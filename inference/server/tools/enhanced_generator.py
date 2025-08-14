"""
Enhanced tool generation with advanced features
"""
import logging
from typing import List, Optional

from .generator import DynamicToolGenerator
from .dynamic_tool import DynamicTool
from .caching import ToolCache
from .learning import ToolLearner
from .composition import ToolComposer

logger = logging.getLogger(__name__)

class EnhancedDynamicToolGenerator(DynamicToolGenerator):
    """Enhanced version with caching, learning, and composition"""

    def __init__(self, llm):
        super().__init__(llm)
        self.tool_cache = ToolCache()
        self.tool_learner = ToolLearner()
        self.tool_composer = ToolComposer()

    async def generate_tool(self, task_description: str) -> Optional[DynamicTool]:
        """Enhanced tool generation with caching and learning"""

        # Check cache first
        cached_tool = self.tool_cache.get_cached_tool(task_description)
        if cached_tool:
            logger.info(f"Using cached tool for: {task_description}")
            return cached_tool

        # Check for tool recommendations
        recommendations = self.tool_learner.get_tool_recommendations(task_description)
        if recommendations:
            logger.info(f"Found recommended tools: {recommendations}")

        # Generate new tool
        tool = await super().generate_tool(task_description)

        if tool:
            # Cache the tool
            self.tool_cache.cache_tool(task_description, tool)
            logger.info(f"Cached new tool: {tool.name}")

        return tool

    async def generate_complex_tool(
        self, task_description: str, subtasks: List[str]
    ) -> Optional[DynamicTool]:
        """Generate a tool that handles complex multi-step tasks"""

        # Generate tools for each subtask
        subtask_tools = []
        for subtask in subtasks:
            tool = await self.generate_tool(subtask)
            if tool:
                subtask_tools.append(tool)

        if len(subtask_tools) == len(subtasks):
            # Create workflow tool
            workflow_tool = self.tool_composer.create_workflow_tool(
                subtask_tools, task_description
            )
            return workflow_tool

        return None
        
    async def generate_parallel_tool(
        self, task_description: str, subtasks: List[str]
    ) -> Optional[DynamicTool]:
        """Generate a tool that handles tasks in parallel"""
        
        # Generate tools for each subtask
        subtask_tools = []
        for subtask in subtasks:
            tool = await self.generate_tool(subtask)
            if tool:
                subtask_tools.append(tool)
                
        if len(subtask_tools) >= 2:  # Need at least 2 tools for parallel execution
            # Create parallel workflow tool
            parallel_tool = self.tool_composer.create_parallel_tool(
                subtask_tools, task_description
            )
            return parallel_tool
            
        # Fall back to sequential if we don't have enough tools
        elif len(subtask_tools) == 1:
            return subtask_tools[0]
            
        return None
