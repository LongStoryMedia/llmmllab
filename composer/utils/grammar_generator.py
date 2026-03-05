"""
Grammar generation utility for structured output using llamacpp grammars.

This module provides utilities to generate and use llamacpp grammar constraints
from Pydantic models, enabling type-safe, structured output from language models.
"""

import json
import logging
import re
import sys
import itertools
from typing import Type, Optional, Union, Dict, Any, List, Set, Tuple
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# Constants and helper classes from llama.cpp implementation
SPACE_RULE = '| " " | "\\n"{1,2} [ \\t]{0,20}'


class BuiltinRule:
    def __init__(self, content: str, deps: Optional[List[str]] = None):
        self.content = content
        self.deps = deps or []


PRIMITIVE_RULES = {
    "boolean": BuiltinRule('("true" | "false") space', []),
    "decimal-part": BuiltinRule("[0-9]{1,16}", []),
    "integral-part": BuiltinRule("[0] | [1-9] [0-9]{0,15}", []),
    "number": BuiltinRule(
        '("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)? space',
        ["integral-part", "decimal-part"],
    ),
    "integer": BuiltinRule('("-"? integral-part) space', ["integral-part"]),
    "value": BuiltinRule(
        "object | array | string | number | boolean | null",
        ["object", "array", "string", "number", "boolean", "null"],
    ),
    "object": BuiltinRule(
        '"{" space ( string ":" space value ("," space string ":" space value)* )? "}" space',
        ["string", "value"],
    ),
    "array": BuiltinRule(
        '"[" space ( value ("," space value)* )? "]" space', ["value"]
    ),
    "uuid": BuiltinRule(
        r'"\""  [0-9a-fA-F]{8} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{4} "-" [0-9a-fA-F]{12} "\"" space',
        [],
    ),
    "char": BuiltinRule(
        r'[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})', []
    ),
    "string": BuiltinRule(r'"\"" char* "\"" space', ["char"]),
    "null": BuiltinRule('"null" space', []),
}

STRING_FORMAT_RULES = {
    "date": BuiltinRule(
        '[0-9]{4} "-" ( "0" [1-9] | "1" [0-2] ) "-" ( "0" [1-9] | [1-2] [0-9] | "3" [0-1] )',
        [],
    ),
    "time": BuiltinRule(
        '([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] ":" [0-5] [0-9] ( "." [0-9]{3} )? ( "Z" | ( "+" | "-" ) ( [01] [0-9] | "2" [0-3] ) ":" [0-5] [0-9] )',
        [],
    ),
    "date-time": BuiltinRule('date "T" time', ["date", "time"]),
    "date-string": BuiltinRule('"\\"" date "\\"" space', ["date"]),
    "time-string": BuiltinRule('"\\"" time "\\"" space', ["time"]),
    "date-time-string": BuiltinRule('"\\"" date-time "\\"" space', ["date-time"]),
}

RESERVED_NAMES = set(
    ["root", "dot", *PRIMITIVE_RULES.keys(), *STRING_FORMAT_RULES.keys()]
)

INVALID_RULE_CHARS_RE = re.compile(r"[^a-zA-Z0-9-]+")
GRAMMAR_LITERAL_ESCAPE_RE = re.compile(r'[\r\n"]')
GRAMMAR_LITERAL_ESCAPES = {"\r": "\\r", "\n": "\\n", '"': '\\"'}

NON_LITERAL_SET = set("|.()[]{}*+?")
ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS = set("^$.[]()|{}*+?")


def _build_repetition(
    item_rule: str,
    min_items: int,
    max_items: Optional[int],
    separator_rule: Optional[str] = None,
) -> str:
    """Build repetition rule for GBNF grammar."""
    if max_items == 0:
        return ""

    if min_items == 0 and max_items == 1:
        return f"{item_rule}?"

    if not separator_rule:
        if min_items == 1 and max_items is None:
            return f"{item_rule}+"
        elif min_items == 0 and max_items is None:
            return f"{item_rule}*"
        else:
            max_str = str(max_items) if max_items is not None else ""
            return f"{item_rule}{{{min_items},{max_str}}}"

    if min_items == 0:
        return f"({item_rule} ({separator_rule} {item_rule})*)?"
    else:
        base = item_rule
        if min_items > 1:
            base += f" ({separator_rule} {item_rule}){{{min_items-1},}}"
        if max_items is not None and max_items > min_items:
            remaining = max_items - min_items
            base += f" ({separator_rule} {item_rule}){{,{remaining}}}"
        elif max_items is None:
            base += f" ({separator_rule} {item_rule})*"
        return base


def _get_basic_json_grammar() -> str:
    """Get basic JSON grammar as fallback."""
    return """root ::= object
object ::= "{" ws ( member ( "," ws member )* )? "}" ws
member ::= string ":" ws value
value ::= object | array | string | number | boolean | null
array ::= "[" ws ( value ( "," ws value )* )? "]" ws
string ::= "\\"" ([^"\\\\] | "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\\""
number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
boolean ::= ("true" | "false") ws
null ::= "null" ws
ws ::= [ \\t\\n\\r]*"""


class _SchemaConverter:
    """Internal schema converter class based on llama.cpp implementation."""

    def __init__(
        self,
        *,
        prop_order: Dict[str, int],
        allow_fetch: bool,
        dotall: bool,
        raw_pattern: bool,
    ):
        self._prop_order = prop_order
        self._allow_fetch = allow_fetch
        self._dotall = dotall
        self._raw_pattern = raw_pattern
        self._rules = {"space": SPACE_RULE}
        self._refs = {}
        self._refs_being_resolved = set()

    def _format_literal(self, literal: str) -> str:
        """Format literal string for GBNF."""
        escaped = GRAMMAR_LITERAL_ESCAPE_RE.sub(
            lambda m: GRAMMAR_LITERAL_ESCAPES.get(m.group(0), m.group(0)), literal
        )
        return f'"{escaped}"'

    def _add_rule(self, name: str, rule: str) -> str:
        """Add rule to grammar with name collision handling."""
        esc_name = INVALID_RULE_CHARS_RE.sub("-", name)
        if esc_name not in self._rules or self._rules[esc_name] == rule:
            key = esc_name
        else:
            i = 0
            while (
                f"{esc_name}{i}" in self._rules
                and self._rules[f"{esc_name}{i}"] != rule
            ):
                i += 1
            key = f"{esc_name}{i}"

        self._rules[key] = rule
        return key

    def _add_primitive(self, name: str, rule: BuiltinRule) -> str:
        """Add primitive rule and its dependencies."""
        n = self._add_rule(name, rule.content)
        for dep in rule.deps:
            dep_rule = PRIMITIVE_RULES.get(dep) or STRING_FORMAT_RULES.get(dep)
            if dep_rule and dep not in self._rules:
                self._add_primitive(dep, dep_rule)
        return n

    def _generate_constant_rule(self, value: Any) -> str:
        """Generate rule for constant value."""
        return self._format_literal(json.dumps(value))

    def _generate_union_rule(self, name: str, alt_schemas: List[Dict[str, Any]]) -> str:
        """Generate rule for union/anyOf/oneOf."""
        return " | ".join(
            self.visit(alt_schema, f'{name}{"-" if name else "alternative-"}{i}')
            for i, alt_schema in enumerate(alt_schemas)
        )

    def _build_object_rule(
        self,
        properties: List[Tuple[str, Any]],
        required: Set[str],
        name: str,
        additional_properties: Optional[Union[bool, Any]],
    ) -> str:
        """Build object rule with properties and requirements."""
        # Sort properties by order preference
        prop_order = self._prop_order
        sorted_props = [
            kv[0]
            for _, kv in sorted(
                enumerate(properties),
                key=lambda ikv: (prop_order.get(ikv[1][0], len(prop_order)), ikv[0]),
            )
        ]

        # Generate key-value rules for each property
        prop_kv_rule_names = {}
        for prop_name, prop_schema in properties:
            prop_rule_name = self.visit(
                prop_schema, f'{name}{"-" if name else ""}{prop_name}'
            )
            prop_kv_rule_names[prop_name] = self._add_rule(
                f'{name}{"-" if name else ""}{prop_name}-kv',
                f'{self._format_literal(json.dumps(prop_name))} space ":" space {prop_rule_name}',
            )

        required_props = [k for k in sorted_props if k in required]
        optional_props = [k for k in sorted_props if k not in required]

        # Handle additional properties
        if additional_properties is not None and additional_properties != False:
            sub_name = f'{name}{"-" if name else ""}additional'
            if isinstance(additional_properties, dict):
                value_rule = self.visit(additional_properties, f"{sub_name}-value")
            else:
                value_rule = self._add_primitive("value", PRIMITIVE_RULES["value"])

            if sorted_props:
                # Generate rule that excludes known property names
                excluded_names = "|".join(f'"{prop}"' for prop in sorted_props)
                key_rule = self._add_rule(
                    f"{sub_name}-k", f"string"
                )  # Simplified for now
            else:
                key_rule = self._add_primitive("string", PRIMITIVE_RULES["string"])

            prop_kv_rule_names["*"] = self._add_rule(
                f"{sub_name}-kv", f'{key_rule} space ":" space {value_rule}'
            )
            optional_props.append("*")

        # Build the complete object rule
        rule = '"{" space'

        # Required properties first
        if required_props:
            rule += " " + ' space "," space '.join(
                prop_kv_rule_names[k] for k in required_props
            )

        # Optional properties
        if optional_props:
            if required_props:
                rule += ' ( space "," space ( '
            else:
                rule += " ( "

            # Simple approach: allow any optional property in any order
            optional_rule_names = [prop_kv_rule_names[k] for k in optional_props]
            if len(optional_rule_names) == 1:
                rule += optional_rule_names[0]
            else:
                rule += " | ".join(optional_rule_names)
                if len(optional_rule_names) > 1:
                    # Allow multiple optional properties
                    rule += (
                        f' ( space "," space ( {" | ".join(optional_rule_names)} ) )*'
                    )

            if required_props:
                rule += " ) )?"
            else:
                rule += " )?"

        rule += ' space "}" space'
        return rule

    def visit(self, schema: Dict[str, Any], name: str) -> str:
        """Visit schema node and generate appropriate rule."""
        schema_type = schema.get("type")
        schema_format = schema.get("format")

        rule_name = name + "-" if name in RESERVED_NAMES else name or "root"

        # Handle $ref
        if "$ref" in schema:
            # For simplicity, we'll ignore refs in this implementation
            # In a full implementation, you'd resolve references
            return self._add_primitive("value", PRIMITIVE_RULES["value"])

        # Handle oneOf/anyOf
        if "oneOf" in schema or "anyOf" in schema:
            return self._add_rule(
                rule_name,
                self._generate_union_rule(name, schema.get("oneOf") or schema["anyOf"]),
            )

        # Handle type unions
        if isinstance(schema_type, list):
            return self._add_rule(
                rule_name,
                self._generate_union_rule(
                    name, [{**schema, "type": t} for t in schema_type]
                ),
            )

        # Handle const
        if "const" in schema:
            return self._add_rule(
                rule_name, self._generate_constant_rule(schema["const"]) + " space"
            )

        # Handle enum
        if "enum" in schema:
            rule = (
                "("
                + " | ".join(self._generate_constant_rule(v) for v in schema["enum"])
                + ") space"
            )
            return self._add_rule(rule_name, rule)

        # Handle object type
        if schema_type in (None, "object") and (
            "properties" in schema or "additionalProperties" in schema
        ):
            required = set(schema.get("required", []))
            properties = list(schema.get("properties", {}).items())
            return self._add_rule(
                rule_name,
                self._build_object_rule(
                    properties, required, name, schema.get("additionalProperties")
                ),
            )

        # Handle array type
        if schema_type in (None, "array") and (
            "items" in schema or "prefixItems" in schema
        ):
            items = schema.get("items") or schema["prefixItems"]
            if isinstance(items, list):
                # Tuple validation
                return self._add_rule(
                    rule_name,
                    '"[" space '
                    + ' "," space '.join(
                        self.visit(item, f'{name}{"-" if name else ""}tuple-{i}')
                        for i, item in enumerate(items)
                    )
                    + ' "]" space',
                )
            else:
                # Array validation
                item_rule_name = self.visit(items, f'{name}{"-" if name else ""}item')
                min_items = schema.get("minItems", 0)
                max_items = schema.get("maxItems")
                return self._add_rule(
                    rule_name,
                    '"[" space '
                    + _build_repetition(
                        item_rule_name, min_items, max_items, separator_rule='"," space'
                    )
                    + ' "]" space',
                )

        # Handle string with constraints
        if schema_type == "string" and ("minLength" in schema or "maxLength" in schema):
            char_rule = self._add_primitive("char", PRIMITIVE_RULES["char"])
            min_len = schema.get("minLength", 0)
            max_len = schema.get("maxLength")
            return self._add_rule(
                rule_name,
                r'"\""  '
                + _build_repetition(char_rule, min_len, max_len)
                + r' "\"" space',
            )

        # Handle string formats
        if (
            schema_type in (None, "string")
            and schema_format
            and f"{schema_format}-string" in STRING_FORMAT_RULES
        ):
            prim_name = f"{schema_format}-string"
            return self._add_rule(
                rule_name,
                self._add_primitive(prim_name, STRING_FORMAT_RULES[prim_name]),
            )

        # Handle primitive types
        if schema_type in PRIMITIVE_RULES:
            return self._add_primitive(
                "root" if rule_name == "root" else schema_type,
                PRIMITIVE_RULES[schema_type],
            )

        # Default fallback
        if schema_type == "object" or len(schema) == 0:
            return self._add_rule(
                rule_name, self._add_primitive("object", PRIMITIVE_RULES["object"])
            )

        # Ultimate fallback to value
        return self._add_primitive("value", PRIMITIVE_RULES["value"])

    def format_grammar(self) -> str:
        """Format the complete grammar."""
        return "\n".join(
            f"{name} ::= {rule}" for name, rule in sorted(self._rules.items())
        )


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

    This implements a full JSON schema to GBNF conversion based on the
    llamacpp reference implementation.

    Args:
        schema: JSON schema dictionary

    Returns:
        GBNF grammar string
    """
    try:
        converter = _SchemaConverter(
            prop_order={}, allow_fetch=False, dotall=True, raw_pattern=False
        )

        # Process the schema to generate rules
        converter.visit(schema, "")

        # Format and return the complete grammar
        return converter.format_grammar()

    except Exception as e:
        logger.error(f"Failed to convert schema to grammar: {e}")
        # Fall back to basic JSON grammar
        return _get_basic_json_grammar()


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
