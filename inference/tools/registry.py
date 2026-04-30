"""
Per-user tool registry — static tools only.

Each user gets a dedicated ToolRegistry that holds instantiated BaseTool
objects. Static tool classes are discovered at import time; instances are
created lazily the first time they are requested.
"""

import asyncio
from typing import Dict, List, Optional

from langchain.tools import BaseTool
from structlog.typing import FilteringBoundLogger

from utils.logging import llmmllogger

# Importing the static tool modules registers them (they decorate BaseTool subclasses).
from tools.static import (  # noqa: F401
    web_search,
    read_web_content,
    memory_retrieval,
    write_todos,
)


class ToolRegistryManager:
    """Manager for per-user ToolRegistry instances."""

    def __init__(self):
        self._user_registries: Dict[str, "ToolRegistry"] = {}
        self._lock = asyncio.Lock()
        self.logger = llmmllogger.logger.bind(component="ToolRegistryManager")

    async def get_user_registry(self, user_id: str) -> "ToolRegistry":
        async with self._lock:
            if user_id not in self._user_registries:
                self.logger.info("Creating new ToolRegistry", user_id=user_id)
                self._user_registries[user_id] = ToolRegistry(user_id=user_id)
            return self._user_registries[user_id]

    def has_user_registry(self, user_id: str) -> bool:
        return user_id in self._user_registries

    async def cleanup_user_registry(self, user_id: str) -> None:
        async with self._lock:
            if user_id in self._user_registries:
                del self._user_registries[user_id]

    async def close(self) -> None:
        async with self._lock:
            self._user_registries.clear()


class ToolRegistry:
    """User-scoped registry for static tools."""

    logger: FilteringBoundLogger

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._tools: Dict[str, BaseTool] = {}
        self._lock = asyncio.Lock()
        self.logger = llmmllogger.logger.bind(component="ToolRegistry", user_id=user_id)
        self._register_statics()

    def _register_statics(self) -> None:
        """Register all static tools available at module import time."""
        # Static tools are module-level tool() instances from tools.static
        for tool_instance in (web_search, read_web_content, memory_retrieval, write_todos):
            self._tools[tool_instance.name] = tool_instance

    def get_all_executable_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_executable_tool(self, tool_name: str) -> Optional[BaseTool]:
        return self._tools.get(tool_name)


registry_manager = ToolRegistryManager()
