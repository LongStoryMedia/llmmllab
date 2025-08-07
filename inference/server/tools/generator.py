"""
Tool generation functionality for creating dynamic tools
"""
import logging
import re
from typing import Dict, Any, Optional

from .dynamic_tool import DynamicTool

logger = logging.getLogger(__name__)


class DynamicToolGenerator:
    """Generates tools dynamically based on LLM requests"""

    def __init__(self, llm):
        self.llm = llm
        self.generated_tools: Dict[str, DynamicTool] = {}

    def generate_tool_prompt(self, task_description: str) -> str:
        """Create a prompt for generating tool code"""
        return f"""
You need to create a Python function to accomplish this task: {task_description}

Please generate Python code that:
1. Defines a single function that accomplishes the task
2. Uses only safe, standard Python libraries (math, datetime, json, re, numpy, pandas if needed)
3. Does not use any file I/O, network operations, or system calls
4. Returns a meaningful result
5. Includes proper error handling

Format your response as:
```python
def function_name(param1, param2, ...):
    \"\"\"
    Brief description of what this function does
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        ...
    
    Returns:
        Description of return value
    \"\"\"
    # Your implementation here
    return result
```

Also provide:
- Tool name (short, descriptive)
- Tool description (one sentence)
- Function name
- Parameter descriptions

Task: {task_description}
"""

    async def generate_tool(self, task_description: str) -> Optional[DynamicTool]:
        """Generate a dynamic tool for the given task"""
        try:
            prompt = self.generate_tool_prompt(task_description)

            # Get response from LLM
            response = await self.llm.generate_async(prompt)

            # Parse the response to extract code and metadata
            tool_info = self._parse_tool_response(response)

            if not tool_info:
                logger.error("Failed to parse tool response")
                return None

            # Create the dynamic tool
            tool = DynamicTool(
                name=tool_info["name"],
                description=tool_info["description"],
                code=tool_info["code"],
                function_name=tool_info["function_name"],
                parameters=tool_info.get("parameters", {}),
            )

            # Store the tool
            self.generated_tools[tool.name] = tool

            logger.info(f"Generated dynamic tool: {tool.name}")
            return tool

        except Exception as e:
            logger.error(f"Error generating dynamic tool: {e}")
            return None

    def _parse_tool_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response to extract tool information"""
        try:
            # Extract code block
            code_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
            if not code_match:
                return None

            code = code_match.group(1).strip()

            # Extract function name from code
            func_match = re.search(r"def\s+(\w+)\s*\(", code)
            if not func_match:
                return None

            function_name = func_match.group(1)

            # Try to extract metadata from response
            lines = response.split("\n")
            tool_name = None
            tool_description = None

            for line in lines:
                if "tool name:" in line.lower():
                    tool_name = line.split(":", 1)[1].strip()
                elif "tool description:" in line.lower():
                    tool_description = line.split(":", 1)[1].strip()

            # Fallback to function name if not found
            if not tool_name:
                tool_name = function_name
            if not tool_description:
                tool_description = f"Dynamically generated tool: {function_name}"

            return {
                "name": tool_name,
                "description": tool_description,
                "code": code,
                "function_name": function_name,
            }

        except Exception as e:
            logger.error(f"Error parsing tool response: {e}")
            return None
