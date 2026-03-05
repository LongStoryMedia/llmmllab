"""
Server interface protocol for composer isolation.

This module defines protocols that composer uses to access server services.
The server implements these protocols to provide data access services.

Composer should NOT access data directly - all data access must go through
the server interface.
"""

from typing import Protocol, Optional, List, Any, Callable
from uuid import UUID
from datetime import datetime

from composer.models import (
    UserConfig,
    Conversation,
    Message,
    Memory,
    MemorySource,
    MessageRole,
    Summary,
    ModelProfile,
    DynamicTool,
    ModelProfileConfig,
)


class UserConfigService(Protocol):
    """Protocol for user config service."""

    async def get_user_config(self, user_id: str) -> UserConfig: ...

    async def update_user_config(self, user_id: str, cfg: dict) -> None: ...

    async def get_all_users(self) -> List[dict]: ...


class ConversationService(Protocol):
    """Protocol for conversation service."""

    async def create_conversation(
        self, conversation: "Conversation"
    ) -> Optional[int]: ...

    async def get_user_conversations(self, user_id: str) -> List[dict]: ...

    async def get_conversation(self, conversation_id: int, user_id: str) -> Conversation: ...

    async def update_conversation_title(
        self, conversation: "Conversation"
    ) -> None: ...

    async def delete_conversation(self, conversation_id: int) -> None: ...


class MessageService(Protocol):
    """Protocol for message service."""

    async def get_conversation_history(
        self, conversation_id: int
    ) -> List[Message]: ...

    async def add_message(self, message: dict, user_config: dict) -> int: ...

    async def get_message(self, message_id: int) -> Optional[dict]: ...

    async def delete_message(self, message_id: int) -> None: ...


class MemoryService(Protocol):
    """Protocol for memory service."""

    async def search_similarity(
        self,
        embeddings: List[List[float]],
        min_similarity: float,
        limit: int,
        user_id: Optional[str],
        conversation_id: Optional[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Memory]: ...

    async def store_memory(
        self,
        user_id: str,
        source: MemorySource,
        role: MessageRole,
        source_id: int,
        embeddings: List[float],
    ) -> None: ...

    async def delete_memory(self, memory_id: str, user_id: str) -> None: ...

    async def delete_all_user_memories(self, user_id: str) -> None: ...


class SummaryService(Protocol):
    """Protocol for summary service."""

    async def create_summary(
        self,
        conversation_id: int,
        content: str,
        level: int,
        source_ids: List[int],
    ) -> int: ...

    async def get_summaries_for_conversation(
        self, conversation_id: int
    ) -> List[Summary]: ...

    async def get_recent_summaries(
        self, conversation_id: int, level: int, limit: int
    ) -> List[Summary]: ...

    async def delete_summaries_for_conversation(self, conversation_id: int) -> None: ...

    async def get_summary(self, summary_id: int) -> Optional[Summary]: ...


class ModelProfileService(Protocol):
    """Protocol for model profile service."""

    async def get_model_profile_by_id(
        self, profile_id: UUID, user_id: str
    ) -> ModelProfile: ...

    async def create_model_profile(self, profile: dict) -> str: ...

    async def update_model_profile(self, profile: dict) -> None: ...

    async def delete_model_profile(self, profile_id: str) -> None: ...

    async def list_model_profiles_by_user(self, user_id: str) -> List[dict]: ...

    async def upsert_default_model_profiles(self) -> None: ...


class DynamicToolService(Protocol):
    """Protocol for dynamic tool service."""

    async def create_tool(self, tool: "DynamicTool") -> DynamicTool: ...

    async def get_tool_by_id(self, tool_id: str) -> Optional[DynamicTool]: ...

    async def get_tools_by_user(self, user_id: str) -> List[DynamicTool]: ...

    async def update_tool(self, tool: "DynamicTool") -> None: ...

    async def delete_tool(self, tool_id: str) -> None: ...


class ServerInterface(Protocol):
    """
    Interface for server services that composer needs.

    This protocol defines all the services composer requires from the server.
    The server implements this interface to provide data access as a gateway.
    """

    user_config: UserConfigService
    conversation: ConversationService
    message: MessageService
    memory: MemoryService
    summary: SummaryService
    model_profile: ModelProfileService
    dynamic_tool: DynamicToolService

    pool: Any
    get_query: Any

    def __init__(self, pool: Any, get_query: Any) -> None:
        ...


class ServerAdapter:
    """
    Adapter to expose server services as ServerInterface for composer.

    This adapter wraps the singleton server instance and provides
    access to all server services through the ServerInterface protocol.
    """

    def __init__(self, server: Optional[Any] = None):
        """
        Initialize the adapter with an optional server instance.

        Args:
            server: Optional server instance. If None, uses the singleton.
        """
        if server is None:
            from composer.server import server as singleton_server  # pylint: disable=import-outside-toplevel
            server = singleton_server
        self._server = server

    @property
    def user_config(self) -> UserConfigService:
        """Get user config service."""
        return self._server.user_config

    @property
    def conversation(self) -> ConversationService:
        """Get conversation service."""
        return self._server.conversation

    @property
    def message(self) -> MessageService:
        """Get message service."""
        return self._server.message

    @property
    def memory(self) -> MemoryService:
        """Get memory service."""
        return self._server.memory

    @property
    def summary(self) -> SummaryService:
        """Get summary service."""
        return self._server.summary

    @property
    def model_profile(self) -> ModelProfileService:
        """Get model profile service."""
        return self._server.model_profile

    @property
    def dynamic_tool(self) -> DynamicToolService:
        """Get dynamic tool service."""
        return self._server.dynamic_tool

    @property
    def pool(self) -> Any:
        """Get database pool."""
        return self._server.pool

    @property
    def get_query(self) -> Any:
        """Get query function."""
        return self._server.get_query