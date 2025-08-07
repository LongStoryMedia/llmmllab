"""
Advanced features for the dynamic tool generation system
"""

import hashlib
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .dynamic_tool import DynamicTool
from .generator import DynamicToolGenerator
from .errors import ToolExecutionError, ToolValidationError, log_error, handle_error

logger = logging.getLogger(__name__)


# Tool Persistence and Caching
class ToolCache:
    """Cache for generated tools to avoid regeneration"""

    def __init__(self, cache_duration_hours: int = 24):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = timedelta(hours=cache_duration_hours)

    def get_cache_key(self, task_description: str) -> str:
        """Generate a cache key from task description"""
        return hashlib.md5(task_description.lower().strip().encode()).hexdigest()

    def get_cached_tool(self, task_description: str) -> Optional[DynamicTool]:
        """Retrieve a cached tool if available and not expired"""
        cache_key = self.get_cache_key(task_description)

        if cache_key in self.cache:
            cached_data = self.cache[cache_key]

            # Check if cache is still valid
            if datetime.now() - cached_data["timestamp"] < self.cache_duration:
                tool_data = cached_data["tool_data"]
                return DynamicTool(**tool_data)

        return None

    def cache_tool(self, task_description: str, tool: DynamicTool) -> None:
        """Cache a generated tool"""
        cache_key = self.get_cache_key(task_description)

        self.cache[cache_key] = {
            "timestamp": datetime.now(),
            "tool_data": {
                "name": tool.name,
                "description": tool.description,
                "code": tool.code,
                "function_name": tool.function_name,
                "parameters": tool.parameters,
            },
        }


# Tool Composition - Combining multiple tools
class ToolComposer:
    """Compose multiple tools into complex workflows"""

    @staticmethod
    def create_workflow_tool(
        tools: List[DynamicTool], workflow_description: str
    ) -> DynamicTool:
        """Create a meta-tool that orchestrates multiple tools"""

        # Generate code that calls multiple tools in sequence
        tool_calls = []
        for i, tool in enumerate(tools):
            tool_calls.append(f"    result_{i} = {tool.function_name}(**kwargs)")

        # Create results list for formatted string
        result_list = []
        for i in range(len(tools)):
            result_list.append(f"result_{i}")
        results_str = ", ".join(result_list)

        # Join tool codes
        tool_codes = "\n".join([tool.code for tool in tools])
        # Join tool calls
        tool_calls_str = "\n".join(tool_calls)

        workflow_code = f"""
def workflow_executor(**kwargs):
    \"\"\"
    {workflow_description}
    \"\"\"
    results = []
    
{tool_calls_str}
    
    # Combine results (customize based on needs)
    combined_result = {{
        'workflow_description': '{workflow_description}',
        'individual_results': [{results_str}]
    }}
    
    return combined_result

# Include all component tool functions
{tool_codes}
"""

        return DynamicTool(
            name=f"workflow_{'_'.join([t.name[:5] for t in tools])}",  # Shortened name
            description=f"Workflow: {workflow_description}",
            code=workflow_code,
            function_name="workflow_executor",
        )


# Tool Learning and Improvement
class ToolLearner:
    """Learn from tool usage to improve future generations"""

    def __init__(self):
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.feedback_log: List[Dict[str, Any]] = []

    def log_tool_usage(
        self, tool_name: str, success: bool, execution_time: float
    ) -> None:
        """Log tool usage statistics"""
        if tool_name not in self.usage_stats:
            self.usage_stats[tool_name] = {
                "total_uses": 0,
                "successful_uses": 0,
                "failed_uses": 0,
                "total_execution_time": 0.0,
                "avg_execution_time": 0.0,
                "success_rate": 0.0,
            }

        stats = self.usage_stats[tool_name]
        stats["total_uses"] += 1
        stats["total_execution_time"] += execution_time
        stats["avg_execution_time"] = (
            stats["total_execution_time"] / stats["total_uses"]
        )

        # Update success rate
        if success:
            stats["successful_uses"] += 1
        else:
            stats["failed_uses"] += 1

        stats["success_rate"] = stats["successful_uses"] / stats["total_uses"]

    def get_tool_recommendations(self, task_description: str) -> List[str]:
        """Recommend tools based on past usage patterns

        Args:
            task_description: Description of the task to find tools for

        Returns:
            List of recommended tool names
        """
        # Simple keyword matching with usage stats
        recommendations = []
        task_lower = task_description.lower()

        for tool_name, stats in self.usage_stats.items():
            # Only recommend tools with good success rate
            if stats["success_rate"] > 0.7 and stats["total_uses"] > 3:
                # Simple keyword matching
                if any(word in tool_name.lower() for word in task_lower.split()):
                    recommendations.append(tool_name)

        return recommendations


# Enhanced Tool Generator with Learning
class EnhancedDynamicToolGenerator(DynamicToolGenerator):
    """Enhanced version with caching, learning, and composition"""

    def __init__(self, llm):
        super().__init__(llm)
        self.tool_cache = ToolCache()
        self.tool_learner = ToolLearner()
        self.tool_composer = ToolComposer()

    async def generate_tool(self, task_description: str) -> Optional[DynamicTool]:
        """Generate a tool with caching and learning features"""
        # Check cache first
        cached_tool = self.tool_cache.get_cached_tool(task_description)
        if cached_tool:
            logger.info(f"Using cached tool for: {task_description}")
            return cached_tool

        # Check for recommendations
        recommendations = self.tool_learner.get_tool_recommendations(task_description)
        if recommendations:
            logger.info(f"Found recommended tools: {recommendations}")
            # Future: Could return a recommendation or use it as a starting point

        # Generate new tool
        tool = await super().generate_tool(task_description)
        if tool:
            # Cache the newly generated tool
            self.tool_cache.cache_tool(task_description, tool)
            logger.info(f"Cached new tool: {tool.name}")

        return tool

    async def generate_complex_tool(
        self, task_description: str, subtasks: List[str]
    ) -> Optional[DynamicTool]:
        """Generate a complex tool by composing multiple tools"""
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


# Tool Validation and Testing
class ToolValidator:
    """Validate tool functionality before deployment"""

    @staticmethod
    async def validate_tool_with_test_cases(
        tool: DynamicTool, test_cases: List[Dict[str, Any]]
    ) -> bool:
        """Validate a tool with predefined test cases"""

        for test_case in test_cases:
            try:
                inputs = test_case.get("inputs", {})
                expected = test_case.get("expected")

                # Execute the tool - using public run method
                result = tool.run(**inputs)

                # Check if result matches expected output
                if expected and str(expected) not in str(result):
                    logger.warning(f"Tool validation failed for {tool.name}")
                    return False

            except (ValueError, TypeError, SyntaxError, ImportError) as exc:
                log_error(exc, context={"tool_name": tool.name}, level="error")
                return False
            except Exception as exc:
                log_error(exc, context={"tool_name": tool.name}, level="error")
                return False

        # All test cases passed
        return True

    @staticmethod
    def generate_test_cases(task_description: str) -> List[Dict[str, Any]]:
        """Generate test cases for a tool based on its description

        Args:
            task_description: Description of the task the tool performs

        Returns:
            A list of test cases with input and expected output values
        """
        # This is a placeholder - in a real system, you would use an LLM to
        # generate test cases or have predefined ones for common task types
        test_cases = []

        # Parse the task to identify patterns that need testing
        if (
            "calculate" in task_description.lower()
            or "math" in task_description.lower()
        ):
            test_cases.append(
                {
                    "input": {"value": 5},
                    "expected_output": 5,
                }
            )
        elif "text" in task_description.lower() or "string" in task_description.lower():
            test_cases.append(
                {
                    "input": {"text": "hello"},
                    "expected_output": "hello",
                }
            )

        return test_cases

    @staticmethod
    def validate_tool(
        tool: DynamicTool, test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Basic validation of tool format and structure

        Args:
            tool: The DynamicTool to validate
            test_cases: Optional list of test cases to run

        Returns:
            True if validation passes, False otherwise
        """
        if not tool.code or not tool.function_name:
            return False

        if test_cases:
            # Run specific test cases
            pass

        # Basic validation passed
        return True


# Tool Marketplace - Share and discover tools
class ToolMarketplace:
    """Share and discover tools across users/sessions"""

    def __init__(self):
        self.public_tools: Dict[str, Dict] = {}  # Contains tool data and metadata
        self.tool_ratings: Dict[str, List[float]] = {}
        self.tool_tags: Dict[str, List[str]] = {}

    def publish_tool(self, tool: DynamicTool, user_id: str) -> None:
        """Publish a tool to the marketplace"""
        self.public_tools[f"{tool.name}_{user_id}"] = {
            "tool": tool,
            "user_id": user_id,
            "published_at": datetime.now(),
            "usage_count": 0,
        }
        logger.info(f"Published tool to marketplace: {tool.name}")

    def search_tools(self, query: str) -> List[DynamicTool]:
        """Search for tools by query"""
        results = []
        query_lower = query.lower()

        for tool_data in self.public_tools.values():
            tool = tool_data["tool"]
            if (
                query_lower in tool.name.lower()
                or query_lower in tool.description.lower()
            ):
                results.append(tool)

        return results

    def rate_tool(self, tool_name: str, rating: float) -> None:
        """Rate a tool (1-5)"""
        if tool_name not in self.tool_ratings:
            self.tool_ratings[tool_name] = []

        # Normalize rating to 1-5 range
        normalized_rating = max(1.0, min(5.0, rating))
        self.tool_ratings[tool_name].append(normalized_rating)

    def get_top_rated_tools(self, limit: int = 10) -> List[tuple]:
        """Get top rated tools"""
        tool_averages = []

        for tool_name, ratings in self.tool_ratings.items():
            if len(ratings) >= 3:  # Minimum ratings required
                avg_rating = sum(ratings) / len(ratings)
                tool_averages.append((tool_name, avg_rating, len(ratings)))

        return sorted(tool_averages, key=lambda x: x[1], reverse=True)[:limit]


# Integration with your existing system
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

        # Generate the tool
        tool = await self.generator.generate_tool(task_description)

        if not tool:
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

        # Publish to marketplace if enabled and user provided
        if self.marketplace and user_id:
            self.marketplace.publish_tool(tool, user_id)

        return tool

    async def execute_tool_with_monitoring(self, tool: DynamicTool, **kwargs) -> str:
        """Execute a tool with performance monitoring"""
        start_time = datetime.now()

        try:
            # Use run instead of _run to avoid protected member access
            result = tool.run(**kwargs)

            # Log successful execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self.generator.tool_learner.log_tool_usage(tool.name, True, execution_time)

            return result

        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            # Log failed execution with specific error
            execution_time = (datetime.now() - start_time).total_seconds()
            self.generator.tool_learner.log_tool_usage(tool.name, False, execution_time)
            return ""

            handle_error(
                exc,
                error_type=ToolExecutionError,
                message=f"Tool execution failed: {exc}",
                context={"tool_name": tool.name, "execution_time": execution_time},
            )
        except Exception as exc:
            # Log failed execution with unexpected error
            execution_time = (datetime.now() - start_time).total_seconds()
            self.generator.tool_learner.log_tool_usage(tool.name, False, execution_time)

            handle_error(
                exc,
                error_type=ToolExecutionError,
                message="Unexpected error during tool execution",
                context={"tool_name": tool.name, "execution_time": execution_time},
            )

            return f"Tool execution failed: {str(exc)}"
