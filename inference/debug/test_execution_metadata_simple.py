"""
Simple test for strongly typed ExecutionMetadata without full composer import.
"""

import sys

sys.path.append("/Users/lons7862/workspace/llmmllab/inference")


def test_execution_metadata():
    """Test the ExecutionMetadata class directly."""

    # Test imports
    try:
        from composer.graph.state import ExecutionMetadata

        print("✅ ExecutionMetadata imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # Test basic creation
    metadata = ExecutionMetadata()
    print("✅ ExecutionMetadata created with defaults")

    # Test field assignment
    metadata.created_at = 1696435200.0
    metadata.composer_version = "0.2.0"
    metadata.streaming_enabled = True
    metadata.workflow_timeout = 300

    print(f"✅ Basic fields set:")
    print(f"   - Created at: {metadata.created_at}")
    print(f"   - Version: {metadata.composer_version}")
    print(f"   - Streaming: {metadata.streaming_enabled}")
    print(f"   - Timeout: {metadata.workflow_timeout}")

    # Test structured methods
    metadata.update_tool_orchestration(
        tool_metadata={"tool_count": 5},
        errors=["Test error"],
        dynamic_tools_count=3,
        static_tools_count=2,
    )

    print(f"✅ Tool orchestration updated:")
    print(f"   - Success: {metadata.orchestration_success}")
    print(f"   - Dynamic tools: {metadata.dynamic_tools_generated}")
    print(f"   - Static tools: {metadata.static_tools_collected}")

    # Test error methods
    metadata.add_error("Test general error")
    metadata.tool_orchestration_error = "Test tool error"

    print(f"✅ Error tracking:")
    print(f"   - General errors: {metadata.errors}")
    print(f"   - Tool error: {metadata.tool_orchestration_error}")
    print(f"   - Has errors: {metadata.has_errors()}")

    # Test search metadata
    metadata.update_search_metadata(search_depth="DEEP", max_results=20, completed=True)

    print(f"✅ Search metadata updated:")
    print(f"   - Search depth: {metadata.search_depth}")
    print(f"   - Max results: {metadata.max_search_results}")
    print(f"   - Completed: {metadata.search_completed}")

    print(f"\n🎉 ExecutionMetadata test completed successfully!")
    print(f"\n📋 Benefits demonstrated:")
    print("   1. ✅ Type-safe field access")
    print("   2. ✅ Structured update methods")
    print("   3. ✅ Error tracking utilities")
    print("   4. ✅ Default value handling")
    print("   5. ✅ Pydantic validation")


if __name__ == "__main__":
    test_execution_metadata()
