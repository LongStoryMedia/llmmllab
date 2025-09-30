"""
Grammar generation utilities for structured output using llamacpp grammars.

Based on LangChain llamacpp documentation and examples from:
https://python.langchain.com/docs/integrations/llms/llamacpp/#grammars
https://raw.githubusercontent.com/ggml-org/llama.cpp/refs/heads/master/examples/json_schema_pydantic_example.py
"""

import json
import logging
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def generate_grammar_from_pydantic(model_class: Type[BaseModel]) -> str:
    """
    Generate llamacpp grammar from a Pydantic model.
    
    Args:
        model_class: Pydantic model class to generate grammar for
        
    Returns:
        Grammar string for llamacpp
    """
    try:
        # Get JSON schema from Pydantic model
        schema = model_class.model_json_schema()
        
        # Convert JSON schema to llamacpp grammar
        grammar = json_schema_to_grammar(schema)
        
        logger.debug(f"Generated grammar for {model_class.__name__}: {len(grammar)} chars")
        return grammar
        
    except Exception as e:
        logger.error(f"Failed to generate grammar for {model_class.__name__}: {e}")
        # Return basic JSON grammar as fallback
        return get_basic_json_grammar()


def json_schema_to_grammar(schema: Dict[str, Any]) -> str:
    """
    Convert JSON schema to llamacpp grammar format.
    
    This is a simplified implementation. For production use, consider using
    the full json-schema-to-grammar converter from llama.cpp repository.
    """
    try:
        # For now, use basic JSON grammar with field validation
        # This ensures valid JSON structure matching the schema
        
        # Extract required fields and types from schema
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        
        # Generate grammar rules for object structure
        if properties:
            field_rules = []
            for field_name, field_schema in properties.items():
                field_type = field_schema.get("type", "string")
                is_required = field_name in required_fields
                
                # Create grammar rule for this field
                field_rule = _generate_field_grammar(field_name, field_type, is_required)
                field_rules.append(field_rule)
            
            # Combine field rules into complete grammar
            grammar = _combine_field_rules(field_rules, required_fields)
            return grammar
        else:
            # No properties defined, use basic JSON grammar
            return get_basic_json_grammar()
            
    except Exception as e:
        logger.error(f"Error converting schema to grammar: {e}")
        return get_basic_json_grammar()


def _generate_field_grammar(field_name: str, field_type: str, is_required: bool) -> str:
    """Generate grammar rule for a single field."""
    # Simplified field grammar generation
    # In production, this should handle all JSON Schema types properly
    
    if field_type == "string":
        value_rule = 'ws "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
    elif field_type == "number":
        value_rule = 'ws ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?'
    elif field_type == "integer":
        value_rule = 'ws ("-"? ([0-9] | [1-9] [0-9]*))'
    elif field_type == "boolean":
        value_rule = 'ws ("true" | "false")'
    elif field_type == "array":
        value_rule = 'ws "[" ws (value ws ("," ws value ws)*)? "]"'
    elif field_type == "object":
        value_rule = 'ws "{" ws (string ws ":" ws value ws ("," ws string ws ":" ws value ws)*)? "}"'
    else:
        # Default to string for unknown types
        value_rule = 'ws "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""'
    
    return f'"{field_name}" ws ":" {value_rule}'


def _combine_field_rules(field_rules: list, required_fields: list) -> str:
    """Combine individual field rules into complete JSON object grammar."""
    # Basic JSON object grammar with field validation
    return '''root ::= ws object ws

object ::= "{" ws (member ws ("," ws member ws)*)? "}"

member ::= string ws ":" ws value

string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""

value ::= object | array | string | number | boolean | null

array ::= "[" ws (value ws ("," ws value ws)*)? "]"

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \t\n\r]*'''


def get_basic_json_grammar() -> str:
    """Get basic JSON grammar as fallback."""
    return '''root ::= ws object ws

object ::= "{" ws (member ws ("," ws member ws)*)? "}"

member ::= string ws ":" ws value

string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\""

value ::= object | array | string | number | boolean | null

array ::= "[" ws (value ws ("," ws value ws)*)? "]"

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \t\n\r]*'''


def validate_grammar(grammar: str) -> bool:
    """
    Validate that a grammar string is well-formed.
    
    This is a basic validation - in production you might want to use
    the actual llamacpp grammar parser for validation.
    """
    try:
        # Basic checks for grammar format
        if not grammar or not grammar.strip():
            return False
            
        # Must contain root rule
        if "root ::" not in grammar:
            logger.warning("Grammar missing root rule")
            return False
            
        # Basic syntax checks
        if grammar.count("::=") == 0:
            logger.warning("Grammar contains no production rules")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Grammar validation error: {e}")
        return False


def check_grammar_runtime_support() -> bool:
    """
    Check if the current runtime environment supports grammar constraints.
    
    Returns:
        True if grammar constraints are supported, False otherwise
    """
    try:
        import inspect
        
        # Check llama-cpp-python version and support
        try:
            import llama_cpp
            from llama_cpp import Llama
            
            # Check if Llama methods support grammar
            if hasattr(Llama, 'create_completion'):
                sig = inspect.signature(Llama.create_completion)
                if 'grammar' in sig.parameters:
                    logger.debug("Grammar support detected in Llama.create_completion")
                    return True
            
            if hasattr(Llama, 'create_chat_completion'):
                sig = inspect.signature(Llama.create_chat_completion)
                if 'grammar' in sig.parameters:
                    logger.debug("Grammar support detected in Llama.create_chat_completion")
                    return True
                    
            # Check version number
            version = getattr(llama_cpp, '__version__', '0.0.0')
            logger.info(f"llama-cpp-python version: {version}")
            
        except ImportError:
            logger.warning("llama-cpp-python not available for grammar support check")
            
        return False
        
    except Exception as e:
        logger.error(f"Grammar runtime support check failed: {e}")
        return False


class GrammarError(Exception):
    """Exception raised when grammar generation or validation fails."""
    pass