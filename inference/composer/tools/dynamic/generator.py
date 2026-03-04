"""
Dynamic Tool implementation for executing custom code at runtime
"""

import io
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from composer.models import DynamicTool
from utils.logging import llmmllogger

from .security import ToolSecurityValidator

logger = llmmllogger.bind(component="dynamic_tool_generator")


class DynamicToolRunner(BaseTool):
    """Dynamically generated tool that can execute custom code"""

    name: str = "dynamic_tool"
    description: str
    code: str
    function_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, dynamic_tool: DynamicTool):
        super().__init__(
            name=dynamic_tool.name,
            description=dynamic_tool.description,
            code=dynamic_tool.code,
            function_name=dynamic_tool.function_name,
            parameters=dynamic_tool.parameters or {},
        )

    def to_tool(self) -> BaseTool:
        """Convert to a LangChain BaseTool for execution"""

        # Capture self references for the closure
        tool_name = self.name
        tool_description = self.description
        tool_runner = self

        def create_tool_function():
            """Create the actual tool function with proper signature"""

            def tool_execution(**kwargs) -> str:
                """Execute the dynamic tool with provided arguments"""
                return tool_runner.execute_tool(**kwargs)

            # Set the docstring dynamically
            tool_execution.__doc__ = (
                f"{tool_description}\n\nDynamically generated tool: {tool_name}"
            )
            tool_execution.__name__ = tool_name

            return tool_execution

        # Create and decorate the function
        tool_func = create_tool_function()
        decorated_tool = tool(tool_func)

        # Update the tool's metadata
        decorated_tool.name = tool_name
        decorated_tool.description = tool_description

        # Set additional attributes to match the original tool properties
        if hasattr(self, "args_schema"):
            decorated_tool.args_schema = self.args_schema
        if hasattr(self, "return_direct"):
            decorated_tool.return_direct = self.return_direct
        if hasattr(self, "tags"):
            decorated_tool.tags = self.tags
        if hasattr(self, "metadata"):
            decorated_tool.metadata = self.metadata
        if hasattr(self, "handle_tool_error"):
            decorated_tool.handle_tool_error = self.handle_tool_error
        if hasattr(self, "handle_validation_error"):
            decorated_tool.handle_validation_error = self.handle_validation_error
        if hasattr(self, "response_format"):
            decorated_tool.response_format = self.response_format

        return decorated_tool

    def execute_tool(self, **kwargs) -> str:
        """Execute the dynamic tool - public interface"""
        return self._run(**kwargs)

    def _run(self, *args, **kwargs) -> str:
        """Execute the dynamic tool"""
        # Extract run_manager if provided, but ignore it for now
        kwargs.pop("run_manager", None)  # Remove run_manager if present

        try:
            # Validate the code before execution
            is_valid, error_msg = ToolSecurityValidator.validate_code(self.code)
            if not is_valid:
                return f"Security validation failed: {error_msg}"

            # Create a restricted execution environment
            restricted_globals = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "reversed": reversed,
                    "any": any,
                    "all": all,
                    "print": print,  # Allow print for debugging
                },
                # Add safe libraries
                "math": __import__("math"),
                "datetime": __import__("datetime"),
                "json": __import__("json"),
                "re": __import__("re"),
                # Add numpy/pandas if available and needed
            }

            try:
                import numpy as np

                restricted_globals["np"] = np
                restricted_globals["numpy"] = np
            except ImportError:
                pass

            try:
                import pandas as pd

                restricted_globals["pd"] = pd
                restricted_globals["pandas"] = pd
            except ImportError:
                pass

            # Capture output
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()

            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                # pylint: disable-next=exec-used
                exec(self.code, restricted_globals)

                # Get the function
                if self.function_name not in restricted_globals:
                    return f"Function '{self.function_name}' not found in executed code"

                func = restricted_globals[self.function_name]

                # Call the function with provided arguments
                result = func(**kwargs)

            # Get any printed output
            printed_output = output_buffer.getvalue()
            error_output = error_buffer.getvalue()

            if error_output:
                return f"Execution error: {error_output}"

            # Format the result
            result_str = str(result) if result is not None else "None"
            if printed_output:
                result_str = f"{printed_output.strip()}\nResult: {result_str}"

            return result_str

        except Exception as e:
            logger.error(f"Error executing dynamic tool: {e}")
            return f"Execution failed: {str(e)}"

    async def execute(self, input_data: Optional[dict] = None, **kwargs) -> str:
        """Public execute method for testing compatibility"""
        if input_data is None:
            input_data = {}
        # Merge input_data with kwargs
        merged_params = {**input_data, **kwargs}
        return self._run(**merged_params)
