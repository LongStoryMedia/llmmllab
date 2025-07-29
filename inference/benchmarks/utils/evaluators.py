"""Evaluator implementation for code execution and validation."""

from typing import Dict, Any
import tempfile
import subprocess
import ast
import os
import re


class AnswerEvaluator:
    """Evaluator for code samples, especially for HumanEval benchmark."""

    def check_code_correctness(
        self, generated_code: str, problem: Dict[str, Any]
    ) -> bool:
        """
        Check if the generated code is correct by running the test cases.

        Args:
            generated_code (str): The generated code solution
            problem (Dict[str, Any]): The problem specification with test cases

        Returns:
            bool: Whether the code passes all test cases
        """
        # Extract function body from the generated code
        fn_body = self._extract_function_body(generated_code, problem)
        if not fn_body:
            return False

        # Create temporary file with test harness
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
            tmp_path = tmp.name
            test_code = self._create_test_harness(fn_body, problem)
            tmp.write(test_code.encode("utf-8"))

        try:
            # Run the test file
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=5,  # Limit execution time to prevent infinite loops
            )

            # Check if the test passed (no error and exit code 0)
            success = result.returncode == 0 and not result.stderr
            return success
        except (subprocess.TimeoutExpired, Exception) as e:
            return False
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_function_body(self, code: str, problem: Dict[str, Any]) -> str:
        """
        Extract the function implementation from generated code.
        """
        # Clean up code to handle potential formatting issues
        # Remove leading and trailing whitespace
        code = code.strip()

        # Try to extract just the function definition without docstring and examples
        function_name = problem["prompt"].split("def ")[1].split("(")[0].strip()

        # Try to find the full function definition
        function_pattern = f"def {function_name}.*?:"
        match = re.search(function_pattern, code, re.DOTALL)

        if not match:
            return ""

        # Find the function signature from the prompt
        prompt_lines = problem["prompt"].strip().split("\n")
        signature_line = [
            line for line in prompt_lines if line.strip().startswith("def ")
        ][0]

        # Combine the signature with the implementation
        try:
            # Parse the code to get the function body
            tree = ast.parse(code)

            # Find the function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get the function body
                    function_body = code[node.body[0].lineno - 1 : node.end_lineno]

                    # Create the full function
                    full_function = signature_line + "\n" + function_body

                    return full_function
        except (SyntaxError, Exception):
            # If parsing fails, try a more basic approach
            lines = code.split("\n")
            body_lines = []
            found_def = False

            for i, line in enumerate(lines):
                if line.strip().startswith(f"def {function_name}"):
                    found_def = True
                    body_lines.append(signature_line)
                elif found_def:
                    # Include the body line if it has content or indentation
                    if line.strip() or line.startswith(" "):
                        body_lines.append(line)
                    else:
                        break  # End of function

            if body_lines:
                return "\n".join(body_lines)

        return ""

    def _create_test_harness(self, function_code: str, problem: Dict[str, Any]) -> str:
        """
        Create a test harness to evaluate the function.
        """
        test_code = problem.get("test", "")

        # Combine the function code with the test
        full_test_code = f"""
{function_code}

{test_code}

# Run the tests
try:
    check({problem['prompt'].split('def ')[1].split('(')[0].strip()})
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")
    exit(1)
except Exception as e:
    print(f"Error: {e}")
    exit(1)
"""
        return full_test_code
