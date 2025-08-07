"""
Dynamic Tool implementation for executing custom code at runtime
"""

import io
import logging
from typing import Any, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import Field

from .security import ToolSecurityValidator

logger = logging.getLogger(__name__)


class DynamicTool(BaseTool):
    """Dynamically generated tool that can execute custom code"""

    name: str = "dynamic_tool"
    description: str
    code: str
    function_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def _run(
        self, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs
    ) -> str:
        """Execute the dynamic tool"""
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
                # Execute the code
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
