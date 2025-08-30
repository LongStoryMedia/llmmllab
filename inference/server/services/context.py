"""
Enhanced conversation context manager with proper async handling and resource management.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Any, Dict
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import HTTPException, Request, status

from runner import pipeline_factory, Embeddings

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    UserConfig,
    Conversation,
    ConversationCtx,
)
from server.auth import get_request_id, get_user_id
from server.utils.chat.message import extract_message_text
from server.config import logger

from .intent import IntentCtx
from ..db import storage
from .search import SearchContext
from .memory import MemoryContext
from .summary import SummaryContext


class ResourceManager:
    """Manages resources with proper cleanup."""

    def __init__(self):
        self._resources: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def add_resource(self, name: str, resource: Any) -> None:
        """Add a resource to be managed."""
        self._resources[name] = resource

    async def cleanup(self) -> None:
        """Clean up all managed resources."""
        for name, resource in self._resources.items():
            try:
                if hasattr(resource, "cleanup"):
                    if asyncio.iscoroutinefunction(resource.cleanup):
                        await resource.cleanup()
                    else:
                        resource.cleanup()
                elif hasattr(resource, "close"):
                    if asyncio.iscoroutinefunction(resource.close):
                        await resource.close()
                    else:
                        resource.close()
            except Exception as e:
                self.logger.error(f"Error cleaning up resource {name}: {e}")

        self._resources.clear()


class ConversationContext(ConversationCtx):
    """
    Enhanced conversation context with proper resource management and concurrent operations.
    """

    def __init__(
        self,
        conversation_id: int,
        user_config: UserConfig,
    ):
        """Initialize conversation context with resource management."""
        current_time = datetime.now()
        user_id = getattr(user_config, "user_id", "default_user")

        # Create placeholder conversation
        placeholder_conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            created_at=current_time,
            updated_at=current_time,
            title="Untitled Conversation",
        )
        self.intent_ctx = IntentCtx()

        super().__init__(
            messages=[],
            notes=[],
            images=[],
            conversation=placeholder_conversation,
            current_user_message=Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text="")],
                conversation_id=conversation_id,
            ),
            intent=self.intent_ctx,
        )

        # Initialize components
        self.user_config = user_config
        self.logger = logging.getLogger("ConversationContext")
        self.resource_manager = ResourceManager()

        # Initialize context services
        self.summary_context = SummaryContext(
            user_cfg=user_config, conversation_id=conversation_id
        )
        self.memory_context = MemoryContext(
            user_cfg=user_config, conversation_id=conversation_id
        )
        self.search_context = SearchContext(user_cfg=user_config)

        # Add resources to manager
        self.resource_manager.add_resource("summary_context", self.summary_context)
        self.resource_manager.add_resource("memory_context", self.memory_context)
        self.resource_manager.add_resource("search_context", self.search_context)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.resource_manager.cleanup()

    def create_system_prompt(
        self, system_prompt_base: str, dynamic_tool_info: str
    ) -> str:
        """Create enhanced system prompt with all context."""
        try:
            prompt = system_prompt_base or "You are a helpful assistant."

            if dynamic_tool_info:
                prompt += f"\n\nAvailable tools:\n{dynamic_tool_info}\n\nUse these tools strategically to provide comprehensive, accurate responses."

            if self.user_config.memory.enabled and hasattr(
                self.memory_context, "memory"
            ):
                memory_content = getattr(self.memory_context, "memory", "")
                if memory_content:
                    prompt += f"\n\nRelevant memories:\n{memory_content}"

            if self.user_config.web_search.enabled and hasattr(
                self.search_context, "research_findings"
            ):
                research_content = getattr(self.search_context, "research_findings", "")
                if research_content:
                    prompt += f"\n\nResearch findings:\n{research_content}\n\nUse these findings to support your responses."

            if self.user_config.summarization.enabled and hasattr(
                self.summary_context, "full_summary"
            ):
                summary_content = getattr(self.summary_context, "full_summary", "")
                if summary_content:
                    prompt += f"\n\nConversation summary:\n{summary_content}"

            return f"{prompt}\n\nAlways explain your reasoning and cite your sources when using retrieved information."

        except Exception as e:
            self.logger.error(f"Error creating system prompt: {e}")
            return system_prompt_base or "You are a helpful assistant."

    async def load_conversation_data(self) -> None:
        """Load conversation data with proper error handling."""
        conversation_id = self.conversation.id

        try:
            # Load conversation details and messages concurrently
            conversation_task = self._load_conversation_details(conversation_id)
            messages_task = self._load_messages(conversation_id)

            conversation_result, messages_result = await asyncio.gather(
                conversation_task, messages_task, return_exceptions=True
            )

            assert conversation_result

            # Handle conversation result
            if isinstance(conversation_result, Exception):
                self.logger.error(f"Error loading conversation: {conversation_result}")
            elif conversation_result:
                assert isinstance(conversation_result, Conversation)
                self.conversation = conversation_result

            # Handle messages result
            if isinstance(messages_result, Exception) or not messages_result:
                self.logger.error(f"Error loading messages: {messages_result}")
                self.messages = []
            else:
                assert isinstance(messages_result, list)
                self.messages = messages_result or []

        except Exception as e:
            self.logger.error(f"Error in load_conversation_data: {e}")

    async def _load_conversation_details(
        self, conversation_id: int
    ) -> Optional[Conversation]:
        """Load conversation details."""
        try:
            return await storage.get_service(storage.conversation).get_conversation(
                conversation_id
            )
        except Exception as e:
            self.logger.error(f"Error loading conversation details: {e}")
            return None

    async def _load_messages(self, conversation_id: int) -> List[Message]:
        """Load conversation messages."""
        try:
            return (
                await storage.get_service(storage.message).get_conversation_history(
                    conversation_id
                )
                or []
            )
        except Exception as e:
            self.logger.error(f"Error loading messages: {e}")
            return []

    async def add_message(self, message: Message) -> Tuple[Embeddings, Optional[int]]:
        """Add message with enhanced error handling and concurrent processing."""
        try:
            # Detect intent for user messages
            if message.role == MessageRole.USER:
                self.current_user_message = message
                if hasattr(self, "intent_ctx"):
                    self.intent = self.intent_ctx.detect(message, self.user_config)

            # Create tasks for concurrent execution
            storage_task = self._store_message(message)
            embedding_task = self._create_embeddings(message)

            # Execute concurrently
            storage_result, embedding_result = await asyncio.gather(
                storage_task, embedding_task, return_exceptions=True
            )

            # Handle storage result
            message_id = None
            if isinstance(storage_result, Exception):
                self.logger.error(f"Error storing message: {storage_result}")
            else:
                message_id = storage_result
                if message_id:
                    assert isinstance(message_id, int)
                    message.id = message_id
                    self.messages.append(message)

            # Handle embedding result
            embeddings = []
            if isinstance(embedding_result, Exception):
                self.logger.error(f"Error creating embeddings: {embedding_result}")
            else:
                embeddings = embedding_result or []

            assert isinstance(embeddings, list)

            return embeddings, message_id

        except Exception as e:
            self.logger.error(f"Error in add_message: {e}")
            return [], None

    async def _store_message(self, message: Message) -> Optional[int]:
        """Store message in database."""
        try:
            return await storage.get_service(storage.message).add_message(message)
        except Exception as e:
            self.logger.error(f"Error storing message: {e}")
            return None

    async def _create_embeddings(self, message: Message) -> List[List[float]]:
        """Create embeddings for message."""
        try:
            text = extract_message_text(message)
            if not text:
                return []

            # Get embedding model profile
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.embedding_profile_id,
                self.user_config.user_id,
            )

            if not mp:
                self.logger.error("Embedding model profile not found")
                return []

            # Get embedding pipeline
            embedding_pipeline = pipeline_factory.get_pipeline(mp, Embeddings)

            # Process message through pipeline
            result = await embedding_pipeline.process_messages([message])

            if isinstance(result, list) and all(
                isinstance(item, list) for item in result
            ):
                return result
            else:
                self.logger.warning(f"Unexpected embedding result type: {type(result)}")
                return []

        except Exception as e:
            self.logger.error(f"Error creating embeddings: {e}")
            return []

    async def generate_title(self) -> str:
        """Generate conversation title with enhanced error handling."""
        try:
            mp = await storage.get_service(
                storage.model_profile
            ).get_model_profile_by_id(
                self.user_config.model_profiles.formatting_profile_id,
                self.user_config.user_id,
            )

            if not mp:
                raise ValueError("Formatting model profile not found")

            # Create title generation prompt
            title_prompt = (
                "Create a short, descriptive title (max 5 words) for this conversation."
            )
            format_message = Message(
                role=MessageRole.USER,
                content=[
                    MessageContent(type=MessageContentType.TEXT, text=title_prompt)
                ],
            )

            # Use the pipeline in a context manager
            with pipeline_factory.pipeline(mp, str) as pipe:
                result = await pipe.process_messages([*self.messages, format_message])

                if isinstance(result, str):
                    title = result.strip()
                    self.conversation.title = title
                    return title

                return "New Conversation"

        except Exception as e:
            self.logger.error(f"Error generating title: {e}")
            return "New Conversation"

    def clear_notes(self) -> None:
        """Clear all notes safely."""
        try:
            self.notes = []
        except Exception as e:
            self.logger.error(f"Error clearing notes: {e}")

    async def process_rag_operations(self, message: Message) -> None:
        """Process RAG operations concurrently with proper error handling."""
        try:
            tasks = []

            # Always run summarization
            if self.user_config.summarization.enabled:
                summarization_task = asyncio.create_task(
                    self.summary_context.summarize(self.messages)
                )
                tasks.append(("summarization", summarization_task))

            assert self.intent, "Intent must be set before processing RAG operations"

            # Conditional memory retrieval
            if self.user_config.memory.enabled and self.intent.memory:
                embeddings, _ = await self.add_message(message)
                if embeddings:
                    memory_task = asyncio.create_task(
                        self.memory_context.retrieve_memories(embeddings)
                    )
                    tasks.append(("memory", memory_task))

            # Conditional web search
            if self.user_config.web_search.enabled and self.intent.web_search:
                search_task = asyncio.create_task(
                    self.search_context.search(message, self.conversation.id)
                )
                tasks.append(("search", search_task))

            if not tasks:
                return

            # Execute all tasks concurrently
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            # Process results
            for (task_name, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in {task_name} task: {result}")
                else:
                    self.logger.debug(f"{task_name} task completed successfully")

        except Exception as e:
            self.logger.error(f"Error processing RAG operations: {e}")


@asynccontextmanager
async def conversation_context_manager(conversation_id: int, user_config: UserConfig):
    """Context manager for conversation contexts."""
    context = ConversationContext(conversation_id, user_config)
    try:
        yield context
    finally:
        await context.resource_manager.cleanup()


async def get_conversation_context_from_request(
    request: Request, conversation_id: int
) -> ConversationContext:
    """
    Enhanced context extraction with proper resource management.
    """
    user_id = get_user_id(request)
    request_id = get_request_id(request)

    if not user_id:
        logger.warning(f"User ID not found for request {request_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated",
        )

    logger.info(f"Processing request {request_id} for user {user_id}")

    try:
        # Get user config
        user_config = await storage.get_service(storage.user_config).get_user_config(
            user_id
        )
        if not user_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User config not found for user {user_id}",
            )

        # Create context
        conversation_ctx = ConversationContext(
            conversation_id=conversation_id,
            user_config=user_config,
        )

        # Load conversation data
        await conversation_ctx.load_conversation_data()

        logger.info(f"Loaded conversation context for conversation {conversation_id}")
        return conversation_ctx

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create conversation context: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation context: {str(e)}",
        ) from e
