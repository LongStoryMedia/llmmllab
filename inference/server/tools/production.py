"""
Production-ready dynamic tool system with all features integrated
"""
import logging
from typing import Optional, List
from datetime import datetime

from .dynamic_tool import DynamicTool
from .enhanced_generator import EnhancedDynamicToolGenerator
from .validation import ToolValidator
from .marketplace import ToolMarketplace

logger = logging.getLogger(__name__)

class ProductionDynamicToolSystem:
    """Production-ready dynamic tool system with all features"""

    def __init__(self, llm, enable_marketplace: bool = False):
        self.generator = EnhancedDynamicToolGenerator(llm)
        self.validator = ToolValidator()
        self.marketplace = ToolMarketplace() if enable_marketplace else None

        # Tool execution monitoring
        self.execution_monitor = {}

    async def create_and_validate_tool(
        self, task_description: str, user_id: Optional[str] = None
    ) -> Optional[DynamicTool]:
        """Create a tool with full validation pipeline"""

        # Check marketplace first
        if self.marketplace and user_id:
            marketplace_tools = self.marketplace.search_tools(task_description)
            if marketplace_tools:
                logger.info(f"Found {len(marketplace_tools)} existing tools in marketplace")
                # Return the first matching tool
                # In a real implementation, you might use a ranking system
                return marketplace_tools[0]

        # Generate the tool
        tool = await self.generator.generate_tool(task_description)

        if not tool:
            logger.warning("Failed to generate tool")
            return None

        # Generate and run test cases
        test_cases = self.validator.generate_test_cases(task_description)
        if test_cases:
            is_valid = await self.validator.validate_tool_with_test_cases(
                tool, test_cases
            )
            if not is_valid:
                logger.warning(f"Tool failed validation: {tool.name}")
                return None

        # Analyze tool complexity
        analysis = self.validator.analyze_tool_complexity(tool)
        logger.info(f"Tool complexity analysis: {analysis}")

        # Publish to marketplace if enabled and user provided
        if self.marketplace and user_id:
            # Extract keywords for tags
            tags = [word.lower() for word in task_description.split() 
                   if len(word) > 4 and word.lower() not in ('this', 'that', 'with', 'from')]
            
            self.marketplace.publish_tool(tool, user_id, tags=tags[:5])  # Limit to 5 tags

        return tool

    async def execute_tool_with_monitoring(self, tool: DynamicTool, **kwargs) -> str:
        """Execute a tool with performance monitoring"""
        start_time = datetime.now()

        try:
            result = tool._run(**kwargs)

            # Log successful execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.generator.tool_learner.log_tool_usage(tool.name, True, execution_time)
            
            # Log marketplace usage if applicable
            if self.marketplace:
                self.marketplace.log_tool_usage(tool.name)
            
            # Store execution metrics
            self.execution_monitor[tool.name] = {
                "last_execution_time": execution_time,
                "last_execution_timestamp": datetime.now(),
                "last_execution_success": True,
                "last_execution_result": result[:100] + "..." if len(result) > 100 else result,
            }

            return result

        except Exception as e:
            # Log failed execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.generator.tool_learner.log_tool_usage(tool.name, False, execution_time)
            
            # Store execution failure metrics
            self.execution_monitor[tool.name] = {
                "last_execution_time": execution_time,
                "last_execution_timestamp": datetime.now(),
                "last_execution_success": False,
                "last_execution_error": str(e),
            }

            return f"Tool execution failed: {str(e)}"
            
    async def create_complex_workflow(
        self, task_description: str, subtasks: List[str], user_id: Optional[str] = None
    ) -> Optional[DynamicTool]:
        """Create a complex workflow tool from multiple subtasks"""
        return await self.generator.generate_complex_tool(task_description, subtasks)
        
    async def create_parallel_workflow(
        self, task_description: str, subtasks: List[str], user_id: Optional[str] = None
    ) -> Optional[DynamicTool]:
        """Create a parallel workflow tool from multiple subtasks"""
        return await self.generator.generate_parallel_tool(task_description, subtasks)
        
    def get_execution_statistics(self):
        """Get statistics about tool executions"""
        if not self.execution_monitor:
            return {"message": "No tool executions recorded yet"}
            
        successful_execs = sum(
            1 for data in self.execution_monitor.values() 
            if data.get("last_execution_success", False)
        )
        
        failed_execs = len(self.execution_monitor) - successful_execs
        
        return {
            "total_executions": len(self.execution_monitor),
            "successful_executions": successful_execs,
            "failed_executions": failed_execs,
            "success_rate": successful_execs / len(self.execution_monitor) if self.execution_monitor else 0,
            "tools_executed": list(self.execution_monitor.keys()),
        }
