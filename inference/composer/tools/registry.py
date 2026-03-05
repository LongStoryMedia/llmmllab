"""
User-aware Tool Registry with per-user tool management and caching.
Supports concurrent users with isolated tool namespaces.
Centralized tool management with sophisticated static/dynamic tool merging.
"""

import asyncio
from typing import Dict, List, Optional, Any, Sequence, TYPE_CHECKING

from structlog.typing import FilteringBoundLogger

from langchain.tools import BaseTool

from composer.models import Tool, UserConfig
from composer.utils.logging import llmmllogger
from composer.tools.static import (
    web_search,
    read_web_content,
    tool_generator,
    write_todos,
)
from composer.agents.engineering_agent import EngineeringAgent

if TYPE_CHECKING:
    from composer.server.interface import ServerInterface


class ToolRegistryManager:
    """
    Manager for per-user ToolRegistry instances.
    Handles creation and caching of user-specific registries.
    """

    def __init__(self):
        self._user_registries: Dict[str, "ToolRegistry"] = {}
        self._lock = asyncio.Lock()
        self.logger = llmmllogger.logger.bind(component="ToolRegistryManager")

    async def get_user_registry(
        self, user_id: str, engineering_agent: Optional[EngineeringAgent] = None
    ) -> "ToolRegistry":
        """Get or create a user-specific ToolRegistry instance."""
        async with self._lock:
            if user_id not in self._user_registries:
                self.logger.info("Creating new ToolRegistry for user", user_id=user_id)
                registry = ToolRegistry(
                    engineering_agent=engineering_agent, user_id=user_id
                )
                self._user_registries[user_id] = registry

            return self._user_registries[user_id]

    def has_user_registry(self, user_id: str) -> bool:
        """Check if a user registry already exists."""
        return user_id in self._user_registries

    async def get_existing_user_registry(
        self, user_id: str
    ) -> Optional["ToolRegistry"]:
        """Get an existing user registry without creating a new one."""
        async with self._lock:
            return self._user_registries.get(user_id)

    async def cleanup_user_registry(self, user_id: str) -> None:
        """Clean up a user's registry when they're done."""
        async with self._lock:
            if user_id in self._user_registries:
                await self._user_registries[user_id].close()
                del self._user_registries[user_id]
                self.logger.info("Cleaned up ToolRegistry for user", user_id=user_id)

    async def close(self) -> None:
        """Clean up all user registries."""
        async with self._lock:
            for registry in self._user_registries.values():
                await registry.close()
            self._user_registries.clear()
            self.logger.info("ToolRegistryManager closed")


class ToolRegistry:
    """
    User-specific registry for static tool management and dynamic tool storage.
    Each user gets their own registry instance with isolated tool caches.
    """

    logger: FilteringBoundLogger

    def __init__(self, engineering_agent: Optional[EngineeringAgent], user_id: str):
        # Static tool classes for instantiation
        self.static_tools: Dict[str, type[BaseTool]] = {}
        # Dynamic tool instances for reuse (tool_id -> Tool)
        self.dynamic_tools: Dict[str, Tool] = {}
        # Executable tool instances (tool_name -> BaseTool instance)
        self.executable_tools: Dict[str, BaseTool] = {}
        # Previous dynamic tools loaded from database (converted to executable tools)
        self.previous_dynamic_tools: Dict[str, BaseTool] = {}

        self.engineering_agent = engineering_agent
        self.user_id = user_id
        self._lock = asyncio.Lock()
        self.logger = llmmllogger.logger.bind(component="ToolRegistry", user_id=user_id)

        self._load_static_tools()

    def _load_static_tools(self):
        """Load static tools from the static tools directory."""
        try:
            tools_to_add = {
                "web_search": web_search,
                "read_web_content": read_web_content,
                "tool_generator": tool_generator,
                "write_todos": write_todos,
                # "memory_retrieval": memory_retrieval,
                # "summarization": summarization,
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
                "Failed to create tool instance",
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

    def get_all_executable_tools(self) -> List[BaseTool]:
        """
        Get all executable tools (static + dynamic) for use by ToolsAgentSubgraph.
        This is the main interface method that replaces the workflow state tool collection.
        """
        all_tools = {}

        # Add static executable tools
        all_tools.update(self.executable_tools)

        # Add previous dynamic tools
        all_tools.update(self.previous_dynamic_tools)

        return list(all_tools.values())

    def convert_tools_to_langchain(self, tools: List[Tool]) -> List[BaseTool]:
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

    async def load_previous_dynamic_tools(
        self, server: Optional["ServerInterface"] = None
    ) -> None:
        """
        Load previously generated dynamic tools from storage and make them available.
        This replaces the functionality of StaticToolLoadingNode.

        Args:
            server: Optional server interface for data access. If None, uses singleton.
        """
        try:
            # Get server - either injected or singleton fallback
            if server is None:
                from composer.server import server  # pylint: disable=import-outside-toplevel

            # Get all previously generated dynamic tools for this user
            dynamic_tools, _ = await server.dynamic_tool.list_tools_by_user(
                user_id=self.user_id, limit=100, offset=0
            )

            # Convert DynamicTool instances to executable tools for reuse
            for dt in dynamic_tools:
                tool_id = f"{self.user_id}_{dt.name}"

                # Create executable tool from the dynamic tool code
                from composer.tools.dynamic.generator import DynamicToolRunner

                executable_tool = DynamicToolRunner(dt).to_tool()

                # Store in previous_dynamic_tools for reuse
                self.previous_dynamic_tools[dt.name] = executable_tool

                # Also register in dynamic_tools for tracking
                tool_model = Tool(
                    name=dt.name,
                    description=dt.description,
                    args_schema=dt.args_schema,
                    return_direct=dt.return_direct,
                    tags=dt.tags,
                    metadata=dt.metadata,
                    handle_tool_error=dt.handle_tool_error,
                    handle_validation_error=dt.handle_validation_error,
                    response_format=dt.response_format,
                )
                self.dynamic_tools[tool_id] = tool_model

            self.logger.info(
                "Loaded previous dynamic tools",
                count=len(self.previous_dynamic_tools),
                tool_names=list(self.previous_dynamic_tools.keys()),
            )

        except Exception as e:
            self.logger.error(f"Failed to load previous dynamic tools: {e}")

    async def generate_new_dynamic_tools(
        self,
        user_query: str,
        user_config: UserConfig,
        server: Optional["ServerInterface"] = None,
    ) -> List[BaseTool]:
        """
        Generate new dynamic tools based on user query and configuration.
        This replaces the functionality of ToolCollectionNode.

        Args:
            user_query: User's request to generate tools for
            user_config: User configuration
            server: Optional server interface for data access. If None, uses singleton.

        Returns:
            List of executable dynamic tools
        """
        try:
            if not user_config.tool.enable_tool_generation:
                self.logger.info("Dynamic tool generation disabled in user config")
                return []

            if not self._should_generate_dynamic_tools(user_query, user_config):
                self.logger.info("Dynamic tool generation not needed based on query")
                return []

            assert self.engineering_agent

            self.logger.info("Generating new dynamic tools based on user query")

            # Get existing static tools to provide context
            static_tools = await self.get_static_tool_instances(self.user_id)

            # Use engineering agent to generate dynamic tool specifications
            dynamic_tool_specs = (
                await self.engineering_agent.generate_dynamic_tool_specification(
                    user_query=user_query,
                    user_id=self.user_id,
                    static_tools=static_tools,
                )
            )

            # Get server - either injected or singleton fallback
            if server is None:
                from composer.server import server  # pylint: disable=import-outside-toplevel

            # Convert specs to executable tools and store
            new_executable_tools = []
            for dt_spec in dynamic_tool_specs:
                # Store in database if storage is available
                dt_spec.user_id = self.user_id
                created_tool = await server.dynamic_tool.create_tool(dt_spec)
                if created_tool:
                    dt_spec = created_tool

                # Create executable tool
                from composer.tools.dynamic.generator import DynamicToolRunner

                executable_tool = DynamicToolRunner(dt_spec).to_tool()
                new_executable_tools.append(executable_tool)

                # Register for tracking
                tool_id = f"{self.user_id}_{dt_spec.name}"
                tool_model = Tool(
                    name=dt_spec.name,
                    description=dt_spec.description,
                    args_schema=dt_spec.args_schema,
                    return_direct=dt_spec.return_direct,
                    tags=dt_spec.tags,
                    metadata=dt_spec.metadata,
                    handle_tool_error=dt_spec.handle_tool_error,
                    handle_validation_error=dt_spec.handle_validation_error,
                    response_format=dt_spec.response_format,
                )
                await self.register_dynamic_tool_instance(
                    tool_id, tool_model, self.user_id
                )

            self.logger.info(
                f"Generated {len(new_executable_tools)} new dynamic tools",
                tool_names=[tool.name for tool in new_executable_tools],
            )

            return new_executable_tools

        except Exception as e:
            self.logger.error(f"Failed to generate dynamic tools: {e}")
            return []

    def _should_generate_dynamic_tools(
        self, user_query: str, user_config: UserConfig
    ) -> bool:
        """
        Determine if dynamic tools should be generated based on user query and configuration.
        """
        # Check user configuration first
        if not user_config.tool.enable_tool_generation:
            return False

        if not self.engineering_agent:
            return False

        # Simple keyword-based heuristic to trigger tool generation
        trigger_keywords = [
            "create a tool",
            "make a function",
            "write a script",
            "generate code to",
            "new tool for",
        ]
        if any(keyword in user_query.lower() for keyword in trigger_keywords):
            return True

        return False

    async def get_all_tools_for_execution(
        self,
        user_query: Optional[str] = None,
        user_config: Optional[UserConfig] = None,
        server: Optional["ServerInterface"] = None,
    ) -> List[BaseTool]:
        """
        Comprehensive tool collection method that replaces all the individual tool nodes.
        Returns deduplicated list of all available executable tools.

        This method:
        1. Gets static tools
        2. Gets previous dynamic tools
        3. Generates new dynamic tools if needed
        4. Deduplicates by name
        5. Returns executable tools ready for use

        Args:
            user_query: Optional user query to generate tools for
            user_config: Optional user configuration
            server: Optional server interface for data access
        """
        async with self._lock:
            all_tools = []
            seen_names = set()

            # 1. Add static executable tools
            for tool_name, tool in self.executable_tools.items():
                if tool_name not in seen_names:
                    all_tools.append(tool)
                    seen_names.add(tool_name)

            # 2. Add previous dynamic tools (converted to executable)
            for tool_name, tool in self.previous_dynamic_tools.items():
                if tool_name not in seen_names:
                    all_tools.append(tool)
                    seen_names.add(tool_name)

            # 3. Generate new dynamic tools if requested and config allows
            if user_query and user_config:
                new_dynamic_tools = await self.generate_new_dynamic_tools(
                    user_query, user_config, server
                )
                for tool in new_dynamic_tools:
                    if tool.name not in seen_names:
                        all_tools.append(tool)
                        seen_names.add(tool.name)

            self.logger.info(
                "Collected all tools for execution",
                total_count=len(all_tools),
                static_count=len(self.executable_tools),
                previous_dynamic_count=len(self.previous_dynamic_tools),
                tool_names=[tool.name for tool in all_tools],
            )

            return all_tools

    async def initialize_for_workflow(
        self,
        user_query: str,
        user_config: UserConfig,
        server: Optional["ServerInterface"] = None,
    ) -> None:
        """
        Initialize the registry with all necessary tools for workflow execution.
        This method replaces the need for separate tool loading/collection/composition nodes.

        Args:
            user_query: User query to initialize tools for
            user_config: User configuration
            server: Optional server interface for data access
        """
        try:
            # Load previous dynamic tools from storage
            await self.load_previous_dynamic_tools(server)

            # Generate new dynamic tools if needed
            new_tools = await self.generate_new_dynamic_tools(
                user_query, user_config, server
            )

            self.logger.info(
                "ToolRegistry initialized for workflow execution",
                static_tools=len(self.executable_tools),
                previous_dynamic_tools=len(self.previous_dynamic_tools),
                new_dynamic_tools=len(new_tools),
                total_tools=len(self.get_all_executable_tools()),
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize ToolRegistry for workflow: {e}")
            raise

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


# Create singleton instance
registry_manager = ToolRegistryManager()
