#!/usr/bin/env python3
"""
Test script to validate the new user config pattern where workflow and tool configs
are always present in user config with proper defaults applied at the storage layer.
"""

import sys
import os

# Add inference path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.default_configs import create_default_user_config
from models.user_config import UserConfig
from models.workflow_config import WorkflowConfig
from models.tool_config import ToolConfig


def test_default_config_creation():
    """Test that default user config always has workflow and tool configs."""
    print("🧪 Testing default user config creation...")
    
    user_config = create_default_user_config("test_user_123")
    
    # Verify basic structure
    assert user_config.user_id == "test_user_123", "User ID should match"
    assert isinstance(user_config.workflow, WorkflowConfig), "Should have WorkflowConfig"
    assert isinstance(user_config.tool, ToolConfig), "Should have ToolConfig"
    
    # Verify workflow defaults
    assert user_config.workflow.enable_streaming == True, "Streaming should be enabled by default"
    assert user_config.workflow.default_timeout == 60.0, "Default timeout should be 60.0"
    assert user_config.workflow.max_parallel_tools == 5, "Default parallel tools should be 5"
    
    # Verify tool defaults
    assert user_config.tool.tool_similarity_threshold == 0.9, "Default similarity threshold should be 0.9"
    assert user_config.tool.enable_tool_generation == True, "Tool generation should be enabled by default"
    assert user_config.tool.tool_cache_ttl == 1800, "Default cache TTL should be 1800"
    
    print("✅ Default config creation test passed!")
    return user_config


def test_custom_config_creation():
    """Test that we can create custom configs and they override defaults properly."""
    print("\n🧪 Testing custom user config creation...")
    
    # Create custom configs
    custom_workflow = WorkflowConfig(
        enable_streaming=False,  # Override default
        default_timeout=120.0,   # Override default
        max_parallel_tools=10    # Override default
        # Other fields will use schema defaults
    )
    
    custom_tool = ToolConfig(
        tool_similarity_threshold=0.7,  # Override default
        enable_tool_generation=False,   # Override default
        tool_timeout=45.0               # Override default
        # Other fields will use schema defaults
    )
    
    # Create user config with custom settings
    user_config = UserConfig(
        user_id="custom_user_456",
        workflow=custom_workflow,
        tool=custom_tool,
        # Use defaults for all other required fields
        preferences=create_default_user_config("temp").preferences,
        memory=create_default_user_config("temp").memory,
        summarization=create_default_user_config("temp").summarization,
        refinement=create_default_user_config("temp").refinement,
        web_search=create_default_user_config("temp").web_search,
        image_generation=create_default_user_config("temp").image_generation,
        model_profiles=create_default_user_config("temp").model_profiles,
        circuit_breaker=create_default_user_config("temp").circuit_breaker,
        gpu_config=create_default_user_config("temp").gpu_config,
    )
    
    # Verify custom overrides
    assert user_config.workflow.enable_streaming == False, "Custom streaming setting should be respected"
    assert user_config.workflow.default_timeout == 120.0, "Custom timeout should be respected"
    assert user_config.workflow.max_parallel_tools == 10, "Custom parallel tools should be respected"
    
    assert user_config.tool.tool_similarity_threshold == 0.7, "Custom similarity threshold should be respected"
    assert user_config.tool.enable_tool_generation == False, "Custom tool generation setting should be respected"
    assert user_config.tool.tool_timeout == 45.0, "Custom tool timeout should be respected"
    
    # Verify defaults are still used for non-overridden fields
    assert user_config.workflow.workflow_cache_ttl == 3600, "Non-overridden fields should use schema defaults"
    assert user_config.tool.tool_cache_ttl == 1800, "Non-overridden fields should use schema defaults"
    
    print("✅ Custom config creation test passed!")
    return user_config


def test_composer_usage_pattern():
    """Test the pattern that composer service would use."""
    print("\n🧪 Testing composer service usage pattern...")
    
    # Simulate what happens in composer service
    user_config = create_default_user_config("composer_test_user")
    
    # This is what the composer service should do instead of fallback to system defaults
    workflow_config = user_config.workflow  # Always present with defaults
    tool_config = user_config.tool          # Always present with defaults
    
    # Verify we can access all needed settings
    streaming_enabled = workflow_config.enable_streaming
    timeout = workflow_config.default_timeout
    max_context_length = workflow_config.max_context_length
    max_parallel_tools = workflow_config.max_parallel_tools
    
    similarity_threshold = tool_config.tool_similarity_threshold
    enable_tool_generation = tool_config.enable_tool_generation
    tool_timeout = tool_config.tool_timeout
    
    # All should be valid values
    assert isinstance(streaming_enabled, bool), "Streaming config should be boolean"
    assert isinstance(timeout, (int, float)) and timeout > 0, "Timeout should be positive number"
    assert isinstance(max_context_length, int) and max_context_length > 0, "Context length should be positive int"
    assert isinstance(max_parallel_tools, int) and max_parallel_tools > 0, "Parallel tools should be positive int"
    
    assert isinstance(similarity_threshold, float) and 0 <= similarity_threshold <= 1, "Similarity threshold should be float 0-1"
    assert isinstance(enable_tool_generation, bool), "Tool generation config should be boolean"
    assert isinstance(tool_timeout, (int, float)) and tool_timeout > 0, "Tool timeout should be positive number"
    
    print("✅ Composer service usage pattern test passed!")
    
    # Show the values for verification
    print(f"📊 Configuration values:")
    print(f"  - Streaming enabled: {streaming_enabled}")
    print(f"  - Workflow timeout: {timeout}s")
    print(f"  - Max context length: {max_context_length}")
    print(f"  - Max parallel tools: {max_parallel_tools}")
    print(f"  - Tool similarity threshold: {similarity_threshold}")
    print(f"  - Tool generation enabled: {enable_tool_generation}")
    print(f"  - Tool timeout: {tool_timeout}s")


def main():
    """Run all validation tests."""
    print("🚀 Starting user config pattern validation tests...")
    print("=" * 60)
    
    try:
        test_default_config_creation()
        test_custom_config_creation()
        test_composer_usage_pattern()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The new user config pattern is working correctly.")
        print("\n📝 Summary:")
        print("  ✅ User configs always have workflow and tool configs with proper defaults")
        print("  ✅ Custom configs properly override defaults while preserving schema defaults")
        print("  ✅ Composer service can access user preferences without fallback logic")
        print("  ✅ Storage layer pattern ensures consistent configuration availability")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()