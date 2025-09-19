"""
Enhanced conversation context manager with proper async handling and resource management.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Any, Dict
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import HTTPException, Request, status
from pydantic import PrivateAttr

from langchain_core.tools import BaseTool
from runner import pipeline_factory, Embeddings
from runner.pipelines.run import run_pipeline, embed_pipeline

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ModelProfile,
    UserConfig,
    Conversation,
    ConversationCtx,
)
from utils.message import extract_message_text

from server.auth import get_request_id, get_user_id
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

    # Private attributes (not part of the Pydantic model state)
    _user_config: UserConfig = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()
    _resource_manager: ResourceManager = PrivateAttr()
    _summary_context: SummaryContext = PrivateAttr()
    _memory_context: MemoryContext = PrivateAttr()
    _search_context: SearchContext = PrivateAttr()

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
        # IMPORTANT: initialize BaseModel first to set up internal pydantic state
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
            intent=IntentCtx(),
        )

        # Initialize components
        self._user_config = user_config
        self._logger = logging.getLogger("ConversationContext")
        self._resource_manager = ResourceManager()

        # Initialize context services
        self._summary_context = SummaryContext(
            user_cfg=user_config, conversation_id=conversation_id
        )
        self._memory_context = MemoryContext(
            user_cfg=user_config, conversation_id=conversation_id
        )
        self._search_context = SearchContext(user_cfg=user_config)

        # Add resources to manager
        self._resource_manager.add_resource("summary_context", self._summary_context)
        self._resource_manager.add_resource("memory_context", self._memory_context)
        self._resource_manager.add_resource("search_context", self._search_context)

    # Public read-only properties for ergonomics
    @property
    def user_config(self) -> UserConfig:
        return self._user_config

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def resource_manager(self) -> ResourceManager:
        return self._resource_manager

    @property
    def summary_context(self) -> SummaryContext:
        return self._summary_context

    @property
    def memory_context(self) -> MemoryContext:
        return self._memory_context

    @property
    def search_context(self) -> SearchContext:
        return self._search_context

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
            # Load conversation details first
            conversation_result = await self._load_conversation_details(conversation_id)

            # Handle conversation result
            if conversation_result:
                assert isinstance(conversation_result, Conversation)
                self.conversation = conversation_result
            else:
                self.logger.error("No conversation found for conversation_id")

            # Load messages sequentially after conversation
            messages_result = await self._load_messages(conversation_id)

            # Handle messages result
            if messages_result:
                assert isinstance(messages_result, list)
                self.messages = messages_result or []
            else:
                self.logger.error("No messages found for conversation")
                self.messages = []

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
            messages = await storage.get_service(
                storage.message
            ).get_conversation_history(conversation_id)
            if not messages:
                self.logger.info(
                    f"No messages found for conversation {conversation_id}"
                )
                return []
            return messages
        except Exception as e:
            self.logger.error(f"Error loading messages: {e}")
            return []

    async def add_message(self, message: Message) -> Tuple[Embeddings, Optional[int]]:
        """Add message with enhanced error handling and concurrent processing."""
        try:
            # Detect intent for user messages
            if message.role == MessageRole.USER:
                self.current_user_message = message
                # Use the model field 'intent' which is an Intent/IntentCtx instance
                if isinstance(self.intent, IntentCtx):
                    self.intent.detect(message, self.user_config)
                else:
                    self.intent = IntentCtx().detect(message, self.user_config)

            # Execute storage and embedding operations concurrently
            storage_task = self._store_message(message)
            embedding_task = self._create_embeddings(message)

            storage_result, embedding_result = await asyncio.gather(
                storage_task, embedding_task, return_exceptions=True
            )

            # Handle storage result
            message_id = None
            if isinstance(storage_result, int):
                message_id = storage_result
                message.id = message_id
                self.messages.append(message)
            elif isinstance(storage_result, Exception):
                self.logger.error(f"Error storing message: {storage_result}")
            else:
                self.logger.error("Error storing message")

            # Handle embedding result
            embeddings = []
            if isinstance(embedding_result, list):
                embeddings = embedding_result
            elif isinstance(embedding_result, Exception):
                self.logger.error(f"Error creating embeddings: {embedding_result}")
            else:
                self.logger.error("Error creating embeddings")

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

            # Get embedding pipeline with HIGH priority (used frequently)
            from runner.pipeline_factory import PipelinePriority

            with pipeline_factory.pipeline(
                mp, Embeddings, PipelinePriority.HIGH
            ) as embedding_pipeline:
                # Process message through normalized embedding interface
                result = await embed_pipeline([message], embedding_pipeline)

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

            # Use the pipeline in a context manager with NORMAL priority (used occasionally)
            from runner.pipeline_factory import PipelinePriority

            with pipeline_factory.pipeline(
                mp,
                str,
                PipelinePriority.NORMAL,
                self.user_config.circuit_breaker,
            ) as pipe:
                response = await run_pipeline([*self.messages, title_prompt], pipe)
                result = (
                    extract_message_text(response.message) if response.message else ""
                )

            if isinstance(result, str) and result.strip():
                title = result.strip()
                self.conversation.title = title
                await storage.get_service(
                    storage.conversation
                ).update_conversation_title(self.conversation.id, title)
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

    def get_enriched_messages(
        self, mp: Optional[ModelProfile], tools: Optional[List[BaseTool]] = None
    ) -> List[Message]:
        """
        Get conversation messages with enriched system prompt including summaries, memory, and search context.
        This method ensures that all conversation context (summaries, memory, search results) is included
        in the system message that gets sent to the pipeline.
        """
        try:
            # Get enhanced system prompt with all context
            tool_info = ""
            if tools:
                tool_descriptions = []
                for tool in tools:
                    tool_descriptions.append(f"- {tool.name}: {tool.description}")
                tool_info = "\n".join(tool_descriptions)

            # Create enhanced system prompt using the conversation context
            enhanced_system_prompt = self.create_system_prompt(
                mp.system_prompt if mp else "You are a helpful assistant.",
                tool_info,
            )

            # Make a copy of messages to avoid modifying the original
            enriched_messages = self.messages.copy()

            # Check if we already have a system message as the first message
            if enriched_messages and enriched_messages[0].role == MessageRole.SYSTEM:
                # Update existing system message with enhanced prompt
                enriched_messages[0] = Message(
                    conversation_id=enriched_messages[0].conversation_id,
                    role=MessageRole.SYSTEM,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=enhanced_system_prompt
                        )
                    ],
                    id=enriched_messages[0].id,
                    created_at=enriched_messages[0].created_at,
                    tool_calls=enriched_messages[0].tool_calls,
                    thinking=enriched_messages[0].thinking,
                )
            else:
                # Insert new system message at the beginning
                system_message = Message(
                    conversation_id=self.conversation.id,
                    role=MessageRole.SYSTEM,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT, text=enhanced_system_prompt
                        )
                    ],
                )
                enriched_messages.insert(0, system_message)

            self.logger.info(
                f"Created enriched message list with {len(enriched_messages)} messages "
                f"(system prompt includes summaries: {bool(self.user_config.summarization.enabled and hasattr(self.summary_context, 'full_summary') and getattr(self.summary_context, 'full_summary', None))}, "
                f"memory: {bool(self.user_config.memory.enabled and hasattr(self.memory_context, 'memory') and getattr(self.memory_context, 'memory', None))}, "
                f"search: {bool(self.user_config.web_search.enabled and hasattr(self.search_context, 'research_findings') and getattr(self.search_context, 'research_findings', None))})"
            )

            self.logger.info(
                f"{'\n'.join([msg.model_dump_json() for msg in enriched_messages])}"
            )
            return enriched_messages

        except Exception as e:
            self.logger.error(f"Error creating enriched messages: {e}")
            # Fallback to original messages if enrichment fails
            return self.messages

    async def process_rag_operations(self, embeddings: List[List[float]]) -> None:
        """Process RAG operations concurrently with proper error handling."""
        try:
            assert self.current_user_message, "Current user message not found"
            assert self.intent, "Intent not set in conversation context"

            query = extract_message_text(self.current_user_message)
            assert query, "Query not found"

            # Create tasks for concurrent execution
            tasks = []

            # Always summarize (lightweight operation)
            tasks.append(self.summary_context.summarize(self.messages))

            # Add conditional tasks based on intent
            if self.intent.memory:
                tasks.append(self.memory_context.retrieve_memories(embeddings))

            if self.intent.web_search:
                tasks.append(
                    self.search_context.search(
                        self.current_user_message, self.conversation.id
                    )
                )

            # Execute all tasks concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

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
