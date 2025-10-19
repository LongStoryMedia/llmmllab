"""
Simplified Tool Registry focusing on static tool management and simple dynamic tool storage.
Removes complex embedding/semantic matching in favor of straightforward tool management.
"""

import asyncio
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from structlog.typing import FilteringBoundLogger

from langchain.tools import BaseTool

from models import Tool
from utils.logging import llmmllogger
from composer.tools.static import (
    memory_retrieval,
    summarization,
)
from composer.tools.static.web_search_tool import web_search

if TYPE_CHECKING:
    from runner import PipelineFactory


class ToolRegistry:
    """
    Simplified registry for static tool management and dynamic tool storage.
    Focuses on clear tool instantiation and storage without complex semantic matching.
    """

    logger: FilteringBoundLogger

    def __init__(self, pipeline_factory: "PipelineFactory"):
        # Static tool classes for instantiation
        self.static_tools: Dict[str, type[BaseTool]] = {}
        # Dynamic tool instances for reuse (tool_id -> Tool)
        self.dynamic_tools: Dict[str, Tool] = {}
        # Executable tool instances (tool_name -> BaseTool instance)
        self.executable_tools: Dict[str, Any] = {}

        self.pipeline_factory = pipeline_factory
        self._lock = asyncio.Lock()
        self.logger = llmmllogger.logger.bind(component="ToolRegistry")

        self._load_static_tools()

    def _load_static_tools(self):
        """Load static tools from the static tools directory."""
        try:
            self.static_tools.update(
                {
                    # "summarization": SummarizationTool,  # Temporarily disabled
                }
            )

            # Add function-based tools that are already decorated with @tool
            self.executable_tools.update(
                {
                    "memory_retrieval": memory_retrieval,
                    "web_search": web_search,
                    "summarization": summarization,
                }
            )

            # Add function-based tools directly to executable_tools
            self.executable_tools["web_search"] = web_search

            self.logger.info(
                "Loaded static tools",
                tool_count=len(self.static_tools),
                executable_count=len(self.executable_tools),
            )

        except ImportError as e:
            self.logger.error(f"Failed to load static tools: {e}")

    async def get_static_tool_instances(self, user_id: str) -> List[Tool]:
        """
        Get instances of all static tools for a user.

        Args:
            user_id: User identifier for configuration

        Returns:
            List of instantiated static Tool objects
        """
        instances = []

        # Handle class-based tools
        for tool_name, tool_cls in self.static_tools.items():
            if tool_cls:
                tool_instance = self._create_tool_instance(tool_cls, user_id)
                if tool_instance:
                    instances.append(tool_instance)

        # Handle function-based tools (with @tool decorator)
        for tool_name, tool_func in self.executable_tools.items():
            if tool_func and hasattr(
                tool_func, "name"
            ):  # Check if it's a @tool decorated function
                tool_instance = Tool(
                    name=getattr(tool_func, "name", tool_name),
                    description=getattr(tool_func, "description", f"{tool_name} tool"),
                    args_schema=getattr(tool_func, "args_schema", None),
                    return_direct=getattr(tool_func, "return_direct", False),
                    tags=getattr(tool_func, "tags", None),
                    metadata=getattr(tool_func, "metadata", None),
                    handle_tool_error=getattr(tool_func, "handle_tool_error", False),
                    handle_validation_error=getattr(
                        tool_func, "handle_validation_error", False
                    ),
                    response_format=getattr(tool_func, "response_format", "content"),
                )
                instances.append(tool_instance)

        return instances

    def _create_tool_instance(self, tool_cls: Any, user_id: str) -> Optional[Tool]:
        """Create tool instance from tool class with user configuration."""
        try:
            # Create tool instance with user_id (class-based tools)
            base_tool = tool_cls(user_id=user_id)

            tool_name = getattr(base_tool, "name", tool_cls.__name__)

            # Store the actual BaseTool instance for execution
            self.executable_tools[tool_name] = base_tool

            # Convert BaseTool instance to our generic Tool model
            tool_instance = Tool(
                name=tool_name,
                description=getattr(
                    base_tool, "description", f"{tool_cls.__name__} tool"
                ),
                args_schema=getattr(base_tool, "args_schema", None),
                return_direct=getattr(base_tool, "return_direct", False),
                tags=getattr(base_tool, "tags", None),
                metadata=getattr(base_tool, "metadata", None),
                handle_tool_error=getattr(base_tool, "handle_tool_error", False),
                handle_validation_error=getattr(
                    base_tool, "handle_validation_error", False
                ),
                response_format=getattr(base_tool, "response_format", "content"),
            )

            self.logger.debug(
                "Created tool instance",
                tool_class=tool_cls.__name__,
                tool_name=tool_name,
                user_id=user_id,
            )
            return tool_instance

        except Exception as e:
            self.logger.error(
                f"Failed to create tool instance",
                tool_class=str(tool_cls),
                user_id=user_id,
                error=str(e),
            )
            return None

    async def register_dynamic_tool_instance(
        self, tool_id: str, tool_instance: Tool, user_id: Optional[str] = None
    ) -> None:
        """
        Register a dynamic tool instance in the registry for potential reuse.

        Args:
            tool_id: Unique identifier for the tool
            tool_instance: The Tool instance to store
            user_id: Optional user id for context
        """
        async with self._lock:
            self.dynamic_tools[tool_id] = tool_instance
            self.logger.info(
                "Registered dynamic tool instance",
                tool_id=tool_id,
                tool_name=tool_instance.name,
                user_id=user_id,
            )

    async def get_dynamic_tool_instances(
        self, user_id: Optional[str] = None
    ) -> List[Tool]:
        """
        Get all dynamic tool instances, optionally filtered by user.

        Args:
            user_id: Optional user identifier for filtering

        Returns:
            List of dynamic Tool instances
        """
        async with self._lock:
            if user_id:
                # Filter by user_id prefix in tool_id
                return [
                    tool
                    for tool_id, tool in self.dynamic_tools.items()
                    if tool_id.startswith(f"{user_id}_")
                ]
            else:
                return list(self.dynamic_tools.values())

    def get_executable_tool(self, tool_name: str) -> Optional[Any]:
        """Get the actual BaseTool instance for execution by tool name."""
        return self.executable_tools.get(tool_name)

    def get_all_executable_tools(self) -> Dict[str, Any]:
        """Get all executable BaseTool instances mapped by name."""
        return self.executable_tools.copy()

    async def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool registry statistics."""
        async with self._lock:
            return {
                "static_tools": len(self.static_tools),
                "dynamic_tools": len(self.dynamic_tools),
                "executable_tools": len(self.executable_tools),
            }

    async def close(self) -> None:
        """Clean up tool registry resources."""
        self.dynamic_tools.clear()
        self.executable_tools.clear()
        self.logger.info("Tool registry closed")
