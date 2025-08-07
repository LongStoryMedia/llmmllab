"""
Security validation for dynamic tools to prevent unsafe code execution
"""
import ast
import logging

logger = logging.getLogger(__name__)


class ToolSecurityValidator:
    """Security validator for dynamically generated tools"""

    FORBIDDEN_IMPORTS = {
        "os",
        "subprocess",
        "sys",
        "shutil",
        "glob",
        "pathlib",
        "socket",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "pickle",
        "marshal",
        "shelve",
        "dbm",
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
    }

    FORBIDDEN_FUNCTIONS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "open",
        "file",
        "input",
        "raw_input",
    }

    FORBIDDEN_ATTRIBUTES = {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__mro__",
        "__globals__",
        "__code__",
        "__func__",
        "__self__",
    }

    @classmethod
    def validate_code(cls, code: str) -> tuple[bool, str]:
        """
        Validate that the generated code is safe to execute

        Args:
            code: The code to validate

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Parse the code into an AST
            tree = ast.parse(code)

            # Walk through all nodes to check for forbidden patterns
            for node in ast.walk(tree):
                # Check for forbidden imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in cls.FORBIDDEN_IMPORTS:
                            return False, f"Forbidden import: {alias.name}"

                elif isinstance(node, ast.ImportFrom):
                    if node.module in cls.FORBIDDEN_IMPORTS:
                        return False, f"Forbidden import from: {node.module}"

                # Check for forbidden function calls
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id in cls.FORBIDDEN_FUNCTIONS
                    ):
                        return False, f"Forbidden function call: {node.func.id}"

                # Check for forbidden attribute access
                elif isinstance(node, ast.Attribute):
                    if node.attr in cls.FORBIDDEN_ATTRIBUTES:
                        return False, f"Forbidden attribute access: {node.attr}"

                # Check for exec/eval in string form
                elif isinstance(node, ast.Str):
                    if any(
                        forbidden in node.s.lower()
                        for forbidden in ["exec(", "eval(", "__import__"]
                    ):
                        return False, "Potential code injection detected"

            return True, "Code validation passed"

        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
