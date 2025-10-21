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
    get_current_date,
    web_search,
)
from composer.tools.utils.schema_filter import patch_tool_schema

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
            # Apply schema filtering to remove InjectedState from tool schemas
            tools_to_add = {
                "memory_retrieval": patch_tool_schema(memory_retrieval),
                "web_search": patch_tool_schema(web_search),
                "summarization": patch_tool_schema(summarization),
                "get_current_date": patch_tool_schema(get_current_date),
            }
            
            self.executable_tools.update(tools_to_add)

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
            List of instantiated static Tool model objects
        """
        instances = []

        # Handle class-based tools
        for tool_name, tool_cls in self.static_tools.items():
            if tool_cls:
                success = self._create_tool_instance(tool_cls, user_id)
                if success and tool_name in self.executable_tools:
                    # Create Tool model object from the executable tool
                    executable_tool = self.executable_tools[tool_name]
                    tool_instance = Tool(
                        name=getattr(executable_tool, "name", tool_name),
                        description=getattr(
                            executable_tool, "description", f"{tool_cls.__name__} tool"
                        ),
                        args_schema=getattr(executable_tool, "args_schema", None),
                        return_direct=getattr(executable_tool, "return_direct", False),
                        tags=getattr(executable_tool, "tags", None),
                        metadata=getattr(executable_tool, "metadata", None),
                        handle_tool_error=getattr(
                            executable_tool, "handle_tool_error", False
                        ),
                        handle_validation_error=getattr(
                            executable_tool, "handle_validation_error", False
                        ),
                        response_format=getattr(
                            executable_tool, "response_format", "content"
                        ),
                    )
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

    def _create_tool_instance(self, tool_cls: Any, user_id: str) -> Optional[bool]:
        """Create tool instance from tool class with user configuration and store in executable_tools."""
        try:
            # Create tool instance with user_id (class-based tools)
            base_tool = tool_cls(user_id=user_id)

            tool_name = getattr(base_tool, "name", tool_cls.__name__)

            # Store the actual BaseTool instance for execution
            self.executable_tools[tool_name] = base_tool

            self.logger.debug(
                "Created tool instance",
                tool_class=tool_cls.__name__,
                tool_name=tool_name,
                user_id=user_id,
            )
            return True  # Just return success flag

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

    def convert_tools_to_langchain(self, tools: List[Tool]) -> List[Any]:
        """
        Convert Tool models to LangChain StructuredTool instances.
        Simplified version - just return the executable tools without LLM-safe wrapper complexity.
        """
        langchain_tools = []

        for tool in tools:
            executable_tool = self.executable_tools.get(tool.name)
            if executable_tool:
                # Just use the tool as-is - no complex wrapper
                langchain_tools.append(executable_tool)
            else:
                self.logger.warning(
                    f"Tool {tool.name} not found in executable tools",
                    tool_name=tool.name,
                )

        return langchain_tools

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
