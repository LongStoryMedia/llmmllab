"""
Simplified ChatService implementation that integrates with the RAG system.
This is a simplified version that removes the complex proto handling to avoid errors.
"""

import time
from datetime import datetime

from models import (
    ChatReq,
    ChatResponse,
    MessageContent,
    MessageContentType,
    Message,
    MessageRole,
)
from ..config import logger
from ..services import LangChainRAGService
from runner.pipelines.factory import PipelineFactory


class RAGChatService:
    """
    Enhanced service for handling chat requests with RAG capabilities.
    This simplified implementation bypasses gRPC complexity.
    """

    def __init__(self):
        """Initialize the RAG chat service."""
        self.logger = logger
        self.pipeline_factory = PipelineFactory()
        self.rag_service = LangChainRAGService()

    def _load_model_if_needed(self, model_id: str):
        """
        Load the model if it's not already loaded using the pipeline factory.

        Args:
            model_id (str): The ID of the model to load.

        Returns:
            Pipeline: The loaded pipeline
        """
        self.logger.info(f"Loading model with ID: {model_id}")

        try:
            # Use the pipeline factory to load the model
            pipeline, load_time = self.pipeline_factory.get_pipeline(model_id)
            self.logger.info(f"Model {model_id} loaded in {load_time:.2f}ms")
            return pipeline
        except (ValueError, RuntimeError) as e:
            self.logger.error(f"Error loading model {model_id}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading model {model_id}: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    async def process_chat_request(
        self, request: ChatReq, user_id: str, conversation_id: int
    ) -> ChatResponse:
        """
        Process a chat request with RAG enhancement.

        Args:
            request: The chat request.
            user_id: The user ID.
            conversation_id: The conversation ID.

        Returns:
            A chat response.
        """
        start_time = time.time()
        self.logger.info(f"Processing chat request for model: {request.model}")

        try:
            # Process with RAG service
            response = await self.rag_service.process_chat_request(
                request, user_id, conversation_id
            )

            processing_time = time.time() - start_time
            self.logger.info(f"Chat request processed in {processing_time:.2f} seconds")

            return response

        except (ValueError, RuntimeError) as e:
            self.logger.error(f"Error processing chat request: {str(e)}")
            # Create and return an error response
            return self._create_error_response(request, str(e))
        except (TypeError, ImportError) as e:
            self.logger.error(f"Type or import error processing chat request: {str(e)}")
            return self._create_error_response(
                request,
                "An error occurred with the processing system. Please try again later.",
            )
        except SystemError as e:
            self.logger.error(f"System error processing chat request: {str(e)}")
            return self._create_error_response(
                request, "A system error occurred. Please try again later."
            )

    def _create_error_response(
        self, request: ChatReq, error_message: str
    ) -> ChatResponse:
        """Create an error response."""
        return ChatResponse(
            model=request.model,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT, text=f"Error: {error_message}"
                    )
                ],
                conversation_id=getattr(request, "conversation_id", -1),
            ),
            created_at=datetime.now(),
            finish_reason="error",
            done=True,
        )
