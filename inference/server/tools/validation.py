"""
Tool validation and testing system to ensure tools work correctly
"""
import logging
from typing import List, Dict, Any

from .dynamic_tool import DynamicTool

logger = logging.getLogger(__name__)

class ToolValidator:
    """Validate tool functionality before deployment"""

    @staticmethod
    async def validate_tool_with_test_cases(
        tool: DynamicTool, test_cases: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate a tool with predefined test cases
        
        Args:
            tool: The dynamic tool to validate
            test_cases: List of test cases with inputs and expected outputs
            
        Returns:
            bool: True if all test cases pass, False otherwise
        """
        for test_case in test_cases:
            try:
                inputs = test_case.get("inputs", {})
                expected = test_case.get("expected")

                # Execute the tool
                result = tool._run(**inputs)

                # Simple validation (you could make this more sophisticated)
                if expected and str(expected) not in str(result):
                    logger.warning(f"Tool validation failed for {tool.name}")
                    return False

            except Exception as e:
                logger.error(f"Tool validation error: {e}")
                return False

        return True

    @staticmethod
    def generate_test_cases(task_description: str) -> List[Dict[str, Any]]:
        """
        Generate test cases for a given task
        
        Args:
            task_description: Description of the task
            
        Returns:
            List[Dict[str, Any]]: List of test cases with inputs and expected outputs
        """
        # This could be enhanced to use LLM to generate test cases
        test_cases = []

        # Basic test case templates based on task type
        lower_task = task_description.lower()
        
        if "calculate" in lower_task or "compute" in lower_task:
            test_cases.append(
                {
                    "inputs": {"x": 10, "y": 5},
                    "expected": None,  # Would need specific validation logic
                }
            )
        elif "convert" in lower_task:
            if "temperature" in lower_task:
                test_cases.append(
                    {
                        "inputs": {"temperature": 32, "from_unit": "fahrenheit", "to_unit": "celsius"},
                        "expected": "0",
                    }
                )
            elif "currency" in lower_task:
                test_cases.append(
                    {
                        "inputs": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
                        "expected": None,
                    }
                )
        elif "sort" in lower_task or "order" in lower_task:
            test_cases.append(
                {
                    "inputs": {"items": [3, 1, 4, 1, 5, 9, 2, 6]},
                    "expected": "[1, 1, 2, 3, 4, 5, 6, 9]",
                }
            )
            
        # Add a default empty test case if no specific ones were generated
        if not test_cases:
            test_cases.append({
                "inputs": {},
                "expected": None
            })

        return test_cases
        
    @staticmethod
    def analyze_tool_complexity(tool: DynamicTool) -> Dict[str, Any]:
        """
        Analyze the complexity and quality of a tool
        
        Args:
            tool: The dynamic tool to analyze
            
        Returns:
            Dict[str, Any]: Analysis results with complexity metrics
        """
        code_lines = tool.code.count('\n')
        has_error_handling = "try:" in tool.code and "except" in tool.code
        has_docstring = '"""' in tool.code or "'''" in tool.code
        has_input_validation = "if " in tool.code and "raise " in tool.code
        
        return {
            "name": tool.name,
            "code_length": len(tool.code),
            "code_lines": code_lines,
            "has_error_handling": has_error_handling,
            "has_docstring": has_docstring,
            "has_input_validation": has_input_validation,
            "complexity_score": (
                (1 if has_error_handling else 0) + 
                (1 if has_docstring else 0) + 
                (1 if has_input_validation else 0)
            ),
        }
