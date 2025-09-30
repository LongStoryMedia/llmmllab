"""
Grammar generation utility for structured output using llamacpp grammars.

This module provides utilities to generate and use llamacpp grammar constraints
from Pydantic models, enabling type-safe, structured output from language models.
"""

import json
import logging
from typing import Type, Optional, Union, Dict, Any
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def pydantic_to_json_schema(model_class: Type[BaseModel]) -> Dict[str, Any]:
    """Convert Pydantic model to JSON schema.

    Args:
        model_class: Pydantic model class

    Returns:
        JSON schema dictionary
    """
    return model_class.model_json_schema()


def json_schema_to_grammar(schema: Dict[str, Any]) -> str:
    """Convert JSON schema to GBNF grammar string.

    This implements a basic JSON schema to GBNF conversion.
    For production use, consider using the llamacpp json_schema_to_grammar.py script.

    Args:
        schema: JSON schema dictionary

    Returns:
        GBNF grammar string
    """
    # Basic GBNF grammar for JSON objects
    # This is a simplified version - for full implementation see:
    # https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/examples/json_schema_to_grammar.py

    grammar_rules = []

    # Basic JSON structure
    grammar_rules.extend(
        [
            "root ::= object",
            'object ::= "{" ws ( member ( "," ws member )* )? "}" ws',
            'member ::= string ":" ws value',
            "value ::= object | array | string | number | boolean | null",
            'array ::= "[" ws ( value ( "," ws value )* )? "]" ws',
            'string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""',
            'number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws',
            'boolean ::= ("true" | "false") ws',
            'null ::= "null" ws',
            "ws ::= [ \\t\\n\\r]*",
        ]
    )

    # TODO: Implement property-specific rules based on schema
    # This would require parsing the schema and generating appropriate GBNF rules
    # For now, we return basic JSON grammar

    return "\n".join(grammar_rules)


def pydantic_to_grammar(model_class: Type[BaseModel]) -> str:
    """Convert Pydantic model directly to GBNF grammar.

    Args:
        model_class: Pydantic model class

    Returns:
        GBNF grammar string
    """
    try:
        # Get JSON schema from Pydantic model
        schema = pydantic_to_json_schema(model_class)

        # Convert to GBNF grammar
        grammar = json_schema_to_grammar(schema)

        logger.debug(
            f"Generated grammar for {model_class.__name__}: {len(grammar)} chars"
        )
        return grammar

    except Exception as e:
        logger.error(f"Error generating grammar for {model_class.__name__}: {e}")
        # Return basic JSON grammar as fallback
        return json_schema_to_grammar({})


def save_grammar_to_file(grammar: str, filepath: Union[str, Path]) -> Path:
    """Save grammar string to .gbnf file.

    Args:
        grammar: GBNF grammar string
        filepath: Path to save the grammar file

    Returns:
        Path to the saved grammar file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(grammar)

    logger.info(f"Grammar saved to {filepath}")
    return filepath


def load_grammar_from_file(filepath: Union[str, Path]) -> str:
    """Load grammar string from .gbnf file.

    Args:
        filepath: Path to the grammar file

    Returns:
        GBNF grammar string
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Grammar file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        grammar = f.read()

    logger.debug(f"Grammar loaded from {filepath}: {len(grammar)} chars")
    return grammar


class GrammarGenerator:
    """Utility class for generating and managing GBNF grammars."""

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize grammar generator with optional caching.

        Args:
            cache_dir: Directory to cache generated grammars
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: Dict[str, str] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_grammar_for_model(
        self, model_class: Type[BaseModel], use_cache: bool = True
    ) -> str:
        """Get GBNF grammar for a Pydantic model with caching.

        Args:
            model_class: Pydantic model class
            use_cache: Whether to use cached grammar if available

        Returns:
            GBNF grammar string
        """
        model_name = model_class.__name__

        # Check memory cache first
        if use_cache and model_name in self._cache:
            logger.debug(f"Using cached grammar for {model_name}")
            return self._cache[model_name]

        # Check file cache
        if use_cache and self.cache_dir:
            cache_file = self.cache_dir / f"{model_name}.gbnf"
            if cache_file.exists():
                try:
                    grammar = load_grammar_from_file(cache_file)
                    self._cache[model_name] = grammar
                    logger.debug(f"Loaded grammar for {model_name} from cache")
                    return grammar
                except Exception as e:
                    logger.warning(
                        f"Failed to load cached grammar for {model_name}: {e}"
                    )

        # Generate new grammar
        grammar = pydantic_to_grammar(model_class)

        # Cache the result
        if use_cache:
            self._cache[model_name] = grammar

            if self.cache_dir:
                try:
                    cache_file = self.cache_dir / f"{model_name}.gbnf"
                    save_grammar_to_file(grammar, cache_file)
                except Exception as e:
                    logger.warning(f"Failed to cache grammar for {model_name}: {e}")

        return grammar

    def clear_cache(self):
        """Clear the in-memory grammar cache."""
        self._cache.clear()
        logger.info("Grammar cache cleared")


# Global instance for convenience
default_generator = GrammarGenerator()


def get_grammar_for_model(model_class: Type[BaseModel], use_cache: bool = True) -> str:
    """Convenience function to get grammar for a Pydantic model.

    Args:
        model_class: Pydantic model class
        use_cache: Whether to use cached grammar if available

    Returns:
        GBNF grammar string
    """
    return default_generator.get_grammar_for_model(model_class, use_cache)


class StructuredOutputError(Exception):
    """Exception raised when structured output parsing fails."""

    pass


def parse_structured_output[T: BaseModel](raw_output: str, model_class: Type[T]) -> T:
    """Parse raw LLM output into a Pydantic model instance.

    Args:
        raw_output: Raw output from the LLM
        model_class: Target Pydantic model class

    Returns:
        Parsed Pydantic model instance

    Raises:
        StructuredOutputError: If parsing fails
    """
    try:
        # Try to parse as JSON first
        if raw_output.strip().startswith("{"):
            data = json.loads(raw_output.strip())
            return model_class.model_validate(data)

        # If not JSON, try to extract JSON from the output
        # Look for JSON-like content between braces
        start_idx = raw_output.find("{")
        end_idx = raw_output.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_content = raw_output[start_idx : end_idx + 1]
            data = json.loads(json_content)
            return model_class.model_validate(data)

        raise StructuredOutputError(
            f"Could not extract valid JSON from output: {raw_output[:100]}..."
        )

    except json.JSONDecodeError as e:
        raise StructuredOutputError(f"JSON parsing failed: {e}")
    except Exception as e:
        raise StructuredOutputError(f"Model validation failed: {e}")
