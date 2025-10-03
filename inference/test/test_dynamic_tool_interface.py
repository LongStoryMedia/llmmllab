"""
Unit tests for DynamicTool BaseTool interface compatibility.

Tests the DynamicTool model's implementation of LangChain BaseTool interface
without requiring external dependencies or infrastructure.
"""

import pytest
import json
from models.dynamic_tool import DynamicTool


class TestDynamicToolBaseTool:
    """Test suite for DynamicTool BaseTool interface implementation."""

    def test_minimal_dynamic_tool_creation(self):
        """Test creating DynamicTool with only required fields."""
        tool = DynamicTool(
            name="test_tool",
            description="A test tool for validation",
            code="def test_func(): return 'test'",
            function_name="test_func",
            user_id="test_user_123",
        )

        # Verify required fields are set
        assert tool.name == "test_tool"
        assert tool.description == "A test tool for validation"
        assert tool.code == "def test_func(): return 'test'"
        assert tool.function_name == "test_func"
        assert tool.user_id == "test_user_123"

        # Verify BaseTool interface defaults
        assert tool.return_direct is False
        assert tool.verbose is False
        assert tool.tags == []
        assert tool.metadata == {}
        assert tool.response_format == "content"
        assert tool.args_schema is None
        assert tool.handle_tool_error is False
        assert tool.handle_validation_error is False

    def test_full_basetool_interface(self):
        """Test creating DynamicTool with all BaseTool interface properties."""
        args_schema = {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
        }

        metadata = {"author": "test", "version": "1.0"}
        tags = ["math", "calculator", "utility"]

        tool = DynamicTool(
            # Database fields
            user_id="test_user_123",
            code="def advanced_func(x, y): return x + y",
            function_name="advanced_func",
            # LangChain BaseTool interface fields
            name="advanced_calculator",
            description="An advanced calculator tool that adds two numbers",
            args_schema=args_schema,
            return_direct=True,
            verbose=True,
            tags=tags,
            metadata=metadata,
            handle_tool_error="Log error and continue",
            handle_validation_error=False,
            response_format="content_and_artifact",
            # Legacy field
            parameters={"legacy_param": "value"},
        )

        # Verify all fields are properly set
        assert tool.name == "advanced_calculator"
        assert tool.description == "An advanced calculator tool that adds two numbers"
        assert tool.args_schema == args_schema
        assert tool.return_direct is True
        assert tool.verbose is True
        assert tool.tags == tags
        assert tool.metadata == metadata
        assert tool.handle_tool_error == "Log error and continue"
        assert tool.handle_validation_error is False
        assert tool.response_format == "content_and_artifact"
        assert tool.parameters == {"legacy_param": "value"}

    def test_response_format_validation(self):
        """Test that response_format accepts valid enum values."""
        # Test default value
        tool = DynamicTool(
            name="test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
        )
        assert tool.response_format == "content"

        # Test explicit valid values
        for format_val in ["content", "content_and_artifact"]:
            tool = DynamicTool(
                name="test",
                description="test",
                code="test",
                function_name="test",
                user_id="test",
                response_format=format_val,
            )
            assert tool.response_format == format_val

    def test_optional_fields_defaults(self):
        """Test that optional fields have proper default values."""
        tool = DynamicTool(
            name="test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
        )

        # Test BaseTool interface defaults
        assert tool.args_schema is None
        assert tool.return_direct is False
        assert tool.verbose is False
        assert tool.tags == []
        assert tool.metadata == {}
        assert tool.handle_tool_error is False
        assert tool.handle_validation_error is False
        assert tool.response_format == "content"

        # Test database layer defaults
        assert tool.id is None
        assert tool.embedding is None
        assert tool.created_at is None
        assert tool.updated_at is None
        assert tool.parameters is None

    def test_model_serialization(self):
        """Test Pydantic model serialization and deserialization."""
        original_tool = DynamicTool(
            name="serialization_test",
            description="Test tool for serialization",
            code="def serialize_test(): return 'serialized'",
            function_name="serialize_test",
            user_id="test_user",
            tags=["test", "serialization"],
            metadata={"test_key": "test_value"},
            args_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            return_direct=True,
            verbose=False,
            response_format="content_and_artifact",
        )

        # Test serialization
        json_data = original_tool.model_dump()

        # Verify all fields are serialized
        expected_fields = {
            "id",
            "user_id",
            "embedding",
            "code",
            "function_name",
            "created_at",
            "updated_at",
            "name",
            "description",
            "args_schema",
            "return_direct",
            "verbose",
            "tags",
            "metadata",
            "handle_tool_error",
            "handle_validation_error",
            "response_format",
            "parameters",
        }
        assert set(json_data.keys()) == expected_fields

        # Test deserialization
        reconstructed_tool = DynamicTool(**json_data)

        # Verify key fields match
        assert original_tool.name == reconstructed_tool.name
        assert original_tool.description == reconstructed_tool.description
        assert original_tool.tags == reconstructed_tool.tags
        assert original_tool.metadata == reconstructed_tool.metadata
        assert original_tool.args_schema == reconstructed_tool.args_schema
        assert original_tool.return_direct == reconstructed_tool.return_direct
        assert original_tool.verbose == reconstructed_tool.verbose
        assert original_tool.response_format == reconstructed_tool.response_format

    def test_json_serializable_fields(self):
        """Test that complex fields can be JSON serialized for database storage."""
        tool = DynamicTool(
            name="json_test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
            args_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            metadata={"key": "value", "nested": {"inner": "data"}},
            tags=["tag1", "tag2"],
        )

        # Test that complex fields are JSON serializable
        assert json.dumps(tool.args_schema) is not None
        assert json.dumps(tool.metadata) is not None
        assert json.dumps(tool.tags) is not None

        # Test JSON roundtrip
        args_json = json.dumps(tool.args_schema)
        metadata_json = json.dumps(tool.metadata)
        tags_json = json.dumps(tool.tags)

        assert json.loads(args_json) == tool.args_schema
        assert json.loads(metadata_json) == tool.metadata
        assert json.loads(tags_json) == tool.tags

    def test_error_handler_field_types(self):
        """Test that error handler fields accept various types as specified."""
        # Test boolean values
        tool_bool = DynamicTool(
            name="test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
            handle_tool_error=True,
            handle_validation_error=False,
        )
        assert tool_bool.handle_tool_error is True
        assert tool_bool.handle_validation_error is False

        # Test string values
        tool_str = DynamicTool(
            name="test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
            handle_tool_error="Log and continue",
            handle_validation_error="Strict validation",
        )
        assert tool_str.handle_tool_error == "Log and continue"
        assert tool_str.handle_validation_error == "Strict validation"

        # Test None values
        tool_none = DynamicTool(
            name="test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
            handle_tool_error=None,
            handle_validation_error=None,
        )
        assert tool_none.handle_tool_error is None
        assert tool_none.handle_validation_error is None

    def test_required_fields_validation(self):
        """Test that required fields are properly validated."""
        required_fields = ["name", "description", "code", "function_name", "user_id"]

        # Valid tool with all required fields
        valid_data = {
            "name": "test",
            "description": "test desc",
            "code": "def test(): pass",
            "function_name": "test",
            "user_id": "user123",
        }

        tool = DynamicTool(**valid_data)
        assert tool is not None

        # Test that missing each required field raises validation error
        for field in required_fields:
            invalid_data = {k: v for k, v in valid_data.items() if k != field}
            with pytest.raises(Exception):  # Pydantic validation error
                DynamicTool(**invalid_data)

    def test_backward_compatibility(self):
        """Test that legacy parameters field is preserved."""
        legacy_params = {"old_param": "old_value", "nested": {"key": "value"}}

        tool = DynamicTool(
            name="legacy_test",
            description="test",
            code="test",
            function_name="test",
            user_id="test",
            parameters=legacy_params,
            # New field alongside legacy
            args_schema={"type": "object"},
        )

        assert tool.parameters == legacy_params
        assert tool.args_schema == {"type": "object"}

        # Both should be serializable
        serialized = tool.model_dump()
        assert serialized["parameters"] == legacy_params
        assert serialized["args_schema"] == {"type": "object"}
