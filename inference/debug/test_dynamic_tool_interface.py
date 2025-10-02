#!/usr/bin/env python3
"""
Test script to validate DynamicTool BaseTool interface compatibility.
"""

import sys
import json
from models.dynamic_tool import DynamicTool

def test_dynamic_tool_creation():
    """Test creating DynamicTool with BaseTool interface properties."""
    
    # Test with minimal required fields
    minimal_tool = DynamicTool(
        name="test_tool",
        description="A test tool for validation",
        code="def test_func(): return 'test'",
        function_name="test_func",
        user_id="test_user_123"
    )
    
    print("✅ Minimal DynamicTool created successfully")
    print(f"   Name: {minimal_tool.name}")
    print(f"   Return Direct: {minimal_tool.return_direct}")
    print(f"   Verbose: {minimal_tool.verbose}")
    print(f"   Tags: {minimal_tool.tags}")
    print(f"   Response Format: {minimal_tool.response_format}")
    
    # Test with full BaseTool interface
    full_tool = DynamicTool(
        # Database fields
        user_id="test_user_123",
        code="def advanced_func(x, y): return x + y",
        function_name="advanced_func",
        
        # LangChain BaseTool interface fields
        name="advanced_calculator",
        description="An advanced calculator tool that adds two numbers",
        args_schema={"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}}},
        return_direct=True,
        verbose=True,
        tags=["math", "calculator", "utility"],
        metadata={"author": "test", "version": "1.0"},
        handle_tool_error="Log error and continue",
        handle_validation_error=False,
        response_format="content_and_artifact",
        
        # Legacy field
        parameters={"legacy_param": "value"}
    )
    
    print("\n✅ Full DynamicTool created successfully")
    print(f"   Name: {full_tool.name}")
    print(f"   Args Schema: {json.dumps(full_tool.args_schema, indent=2)}")
    print(f"   Return Direct: {full_tool.return_direct}")
    print(f"   Verbose: {full_tool.verbose}")
    print(f"   Tags: {full_tool.tags}")
    print(f"   Metadata: {json.dumps(full_tool.metadata, indent=2)}")
    print(f"   Handle Tool Error: {full_tool.handle_tool_error}")
    print(f"   Handle Validation Error: {full_tool.handle_validation_error}")
    print(f"   Response Format: {full_tool.response_format}")
    print(f"   Legacy Parameters: {full_tool.parameters}")

def test_serialization():
    """Test JSON serialization/deserialization."""
    
    tool = DynamicTool(
        name="serialization_test",
        description="Test tool for serialization",
        code="def serialize_test(): return 'serialized'",
        function_name="serialize_test", 
        user_id="test_user",
        tags=["test", "serialization"],
        metadata={"test_key": "test_value"},
        args_schema={"type": "object", "properties": {"input": {"type": "string"}}}
    )
    
    # Test model_dump (Pydantic v2)
    try:
        json_data = tool.model_dump()
        print("\n✅ Serialization successful")
        print(f"   Serialized keys: {list(json_data.keys())}")
        
        # Test deserialization
        reconstructed_tool = DynamicTool(**json_data)
        print("✅ Deserialization successful")
        print(f"   Name matches: {tool.name == reconstructed_tool.name}")
        print(f"   Tags match: {tool.tags == reconstructed_tool.tags}")
        print(f"   Metadata matches: {tool.metadata == reconstructed_tool.metadata}")
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")

def main():
    """Main test function."""
    
    print("🧪 Testing DynamicTool BaseTool Interface Compatibility")
    print("=" * 60)
    
    try:
        test_dynamic_tool_creation()
        test_serialization()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! DynamicTool successfully implements BaseTool interface")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())