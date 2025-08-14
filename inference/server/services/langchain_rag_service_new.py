"""
Integration of the RAG system with the Chat service.
This file provides the LangChain implementation that combines all components.
"""

import asyncio
from typing import List
from datetime import datetime

# Import LangChain components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

# Import local models
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    ChatReq,
    ChatResponse,
)

# Import local pipeline factory
from runner.pipelines.factory import PipelineFactory

import server.config

logger = server.config.logger  # Use the logger from config


class LangChainRAGService:
    """
    Service that integrates LangChain components for RAG functionality.
    Uses the local model pipeline for LLM interactions and enhances
    responses with relevant context from vector stores and memory.
    """

    def __init__(self):
        """Initialize the LangChain RAG service."""
        # Initialize the pipeline factory
        self.pipeline_factory = PipelineFactory()

        # Initialize embeddings model
        self.embeddings = self._initialize_embeddings()

        # Create the vector store
        self.vector_store = self._initialize_vector_store()

        # Initialize memory cache
        self.memory_cache = {}

        # Initialize the text splitter for documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

    def _initialize_embeddings(self):
        """
        Initialize the embeddings model for vector operations.
        Defaults to a local HuggingFace model for efficiency.
        """
        try:
            # Use sentence-transformers model for embeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except (ImportError, ValueError) as e:
            logger.error(f"Error initializing embeddings: {e}")
            # Fallback to a simpler model if necessary
            return HuggingFaceEmbeddings(model_name="distilbert-base-uncased")
        except (RuntimeError, OSError) as e:
            logger.error(f"Runtime error initializing embeddings: {e}")
            # Fallback to a basic model as last resort
            return HuggingFaceEmbeddings(model_name="distilbert-base-uncased")

    def _initialize_vector_store(self):
        """
        Initialize the vector store for document storage and retrieval.
        Uses in-memory FAISS for efficiency.
        """
        try:
            # Create empty FAISS vector store
            return FAISS.from_documents(
                documents=[
                    Document(
                        page_content="Initial document", metadata={"source": "init"}
                    )
                ],
                embedding=self.embeddings,
            )
        except (ImportError, ValueError) as e:
            logger.error(f"Error initializing vector store: {e}")
            return None
        except (RuntimeError, AttributeError) as e:
            logger.error(f"Runtime error initializing vector store: {e}")
            return None

    def _get_memory_key(self, user_id: str, conversation_id: int) -> str:
        """Generate a unique key for memory cache."""
        return f"{user_id}_{conversation_id}"

    def _get_conversation_memory(
        self, user_id: str, conversation_id: int
    ) -> ConversationBufferMemory:
        """
        Get or create conversation memory for a specific user and conversation.

        Args:
            user_id: The user ID.
            conversation_id: The conversation ID.

        Returns:
            A ConversationBufferMemory instance.
        """
        memory_key = self._get_memory_key(user_id, conversation_id)

        if memory_key not in self.memory_cache:
            # Create new memory
            self.memory_cache[memory_key] = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            )

        return self.memory_cache[memory_key]

    async def _add_documents_to_vector_store(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store for retrieval.

        Args:
            documents: List of documents to add.
        """
        if not self.vector_store or not documents:
            return

        try:
            # Process in batches to avoid memory issues
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                self.vector_store.add_documents(batch)

            logger.info(f"Added {len(documents)} documents to vector store")
        except ValueError as e:
            logger.error(f"Value error adding documents to vector store: {e}")
        except AttributeError as e:
            logger.error(f"Vector store attribute error: {e}")
        except (RuntimeError, OSError, IOError) as e:
            logger.error(f"Error adding documents to vector store: {e}")

    async def process_chat_request(
        self, request: ChatReq, user_id: str, conversation_id: int
    ) -> ChatResponse:
        """
        Process a chat request with RAG enhancements.

        Args:
            request: The chat request.
            user_id: The user ID.
            conversation_id: The conversation ID.

        Returns:
            A chat response.
        """
        # Extract the user message
        user_message = None
        for msg in request.messages:
            if msg.role == MessageRole.USER:
                user_message = msg
                break

        if not user_message:
            # No user message found
            return self._create_error_response(
                request, "I couldn't find a user message to respond to."
            )

        # Extract user query from message
        user_query = ""
        for content in user_message.content:
            if content.type == MessageContentType.TEXT and content.text:
                user_query += content.text + " "

        user_query = user_query.strip()
        if not user_query:
            return self._create_error_response(
                request, "I couldn't find any text in your message."
            )

        # Get conversation memory
        memory = self._get_conversation_memory(user_id, conversation_id)

        try:
            # Create the RAG chain
            response_text = await self._run_rag_chain(
                user_query=user_query,
                model_id=request.model,
                memory=memory,
                conversation_id=conversation_id,
                user_id=user_id,
                options=request.options,
            )

            # Create the response
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=response_text)
                ],
                conversation_id=conversation_id,
                created_at=datetime.now(),
            )

            # Update memory with this exchange
            memory.save_context({"input": user_query}, {"output": response_text})

            # Create and return the response
            return ChatResponse(
                model=request.model,
                message=assistant_message,
                created_at=datetime.now(),
                finish_reason="stop",
                done=True,  # Indicate that the generation is complete
            )

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error processing chat request: {e}")
            return self._create_error_response(
                request,
                "I encountered an error while processing your request. Please try again.",
            )
        except KeyError as e:
            # Log key errors
            logger.error(f"Key error in process_chat_request: {e}")
            return self._create_error_response(
                request, "I encountered a data access error. Please try again."
            )
        except AttributeError as e:
            # Log attribute errors
            logger.error(f"Attribute error in process_chat_request: {e}")
            return self._create_error_response(
                request, "I encountered a data processing error. Please try again."
            )
        except (IOError, OSError) as e:
            # Log I/O errors
            logger.error(f"I/O error in process_chat_request: {e}")
            return self._create_error_response(
                request, "I encountered a system error. Please try again later."
            )

    async def _run_rag_chain(
        self,
        user_query: str,
        model_id: str,
        # Parameters below are kept for API compatibility but not used in current implementation
        # Will be useful for future extensions
        memory=None,  # pylint: disable=unused-argument
        conversation_id=None,  # pylint: disable=unused-argument
        user_id=None,  # pylint: disable=unused-argument
        options=None,  # pylint: disable=unused-argument
    ) -> str:
        """
        Run the RAG chain to generate a response.

        Args:
            user_query: The user's query.
            model_id: The model ID to use.
            memory: Placeholder for future memory integration.
            conversation_id: Placeholder for future conversation tracking.
            user_id: Placeholder for future user tracking.
            options: Placeholder for future model parameters.

        Returns:
            The generated response text.
        """
        # Get the model through the pipeline factory
        try:
            # Load the model with pipeline factory
            pipeline, load_time = self.pipeline_factory.get_pipeline(model_id)
            logger.info(f"Model {model_id} loaded in {load_time:.2f}ms")

            # If we have a retriever, use a RAG chain
            if self.vector_store:
                # Get relevant documents
                docs = self.vector_store.similarity_search(user_query, k=3)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Format prompt with context
                formatted_prompt = (
                    "Context: "
                    + context
                    + "\n\nQuestion: "
                    + user_query
                    + "\n\nAnswer:"
                )
            else:
                # If no retriever, just use the model directly
                formatted_prompt = "Question: " + user_query + "\n\nAnswer:"

            # Use pipeline to generate response
            return await asyncio.to_thread(
                lambda: self._run_pipeline(pipeline, formatted_prompt, load_time)
            )

        except (ValueError, RuntimeError) as e:
            logger.error(f"Error in RAG chain: {e}")
            # Fallback to simple response
            return "I apologize, but I encountered an error while processing your request. Please try again."
        except (TypeError, KeyError, AssertionError) as e:
            # Log specific errors
            logger.error(f"Error in _run_rag_chain: {e}")
            return "An error occurred while processing your request. Please try again later."

    def _run_pipeline(self, pipeline, prompt_str: str, load_time: float) -> str:
        """Run the model pipeline with the given prompt."""
        # Create a chat request with the formatted prompt
        request = ChatReq(
            model=pipeline.model_def.id,
            messages=[
                Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(type=MessageContentType.TEXT, text=prompt_str)
                    ],
                    conversation_id=-1,
                )
            ],
            stream=False,
        )

        # Run the pipeline and extract the response
        result = pipeline.run(request, load_time)

        # Extract the text response based on the result type
        if isinstance(result, str):
            return result
        elif hasattr(result, "message") and hasattr(result.message, "content"):
            # Extract text from message content
            for content in result.message.content:
                if content.type == MessageContentType.TEXT and content.text:
                    return content.text
            return "No text response generated."
        else:
            return str(result)

    def _create_error_response(
        self, request: ChatReq, error_message: str
    ) -> ChatResponse:
        """Create an error response."""
        return ChatResponse(
            model=request.model,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=error_message)
                ],
                conversation_id=(
                    request.conversation_id
                    if hasattr(request, "conversation_id")
                    and request.conversation_id is not None
                    else -1
                ),
            ),
            created_at=datetime.now(),
            finish_reason="error",
            done=True,  # Indicate that the generation is complete
        )
