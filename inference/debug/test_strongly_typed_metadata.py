"""
Demonstration of strongly typed ExecutionMetadata benefits.
Shows the improvements in type safety, validation, and API usability.
"""


def demonstrate_strongly_typed_execution_metadata():
    """Demonstrate the benefits of strongly typed ExecutionMetadata."""

    print("🔧 Demonstrating Strongly Typed ExecutionMetadata")
    print("=" * 55)

    # Import the new strongly typed model
    import sys

    sys.path.append("/Users/lons7862/workspace/llmmllab/inference")

    from composer.graph.state import ExecutionMetadata, WorkflowState
    from models.lang_chain_message import LangChainMessage

    print("\n📋 Test Case 1: Type-Safe Creation and Access")
    print("-" * 45)

    # Create strongly typed metadata
    metadata = ExecutionMetadata(
        created_at=1696435200.0,
        composer_version="0.2.0",
        streaming_enabled=True,
        workflow_timeout=300,
    )

    print(f"✅ Created metadata with type safety:")
    print(f"   - Created at: {metadata.created_at}")
    print(f"   - Version: {metadata.composer_version}")
    print(f"   - Streaming: {metadata.streaming_enabled}")
    print(f"   - Timeout: {metadata.workflow_timeout}")

    print(f"\n📋 Test Case 2: Structured Update Methods")
    print("-" * 45)

    # Use structured update methods
    metadata.update_tool_orchestration(
        tool_metadata={"tool_count": 5, "generation_time": 1.2},
        errors=["Tool X failed to generate"],
        dynamic_tools_count=3,
        static_tools_count=2,
    )

    print(f"✅ Updated tool orchestration metadata:")
    print(f"   - Tool metadata: {metadata.tool_orchestration}")
    print(f"   - Errors: {metadata.tool_generation_errors}")
    print(f"   - Success: {metadata.orchestration_success}")
    print(f"   - Dynamic tools: {metadata.dynamic_tools_generated}")
    print(f"   - Static tools: {metadata.static_tools_collected}")

    metadata.update_search_metadata(
        search_depth="DEEP",
        max_results=20,
        search_method="web_search",
        results={"total_found": 15, "relevant": 12},
        completed=True,
    )

    print(f"✅ Updated search metadata:")
    print(f"   - Search depth: {metadata.search_depth}")
    print(f"   - Max results: {metadata.max_search_results}")
    print(f"   - Method: {metadata.search_method}")
    print(f"   - Results: {metadata.web_search_results}")
    print(f"   - Completed: {metadata.search_completed}")

    print(f"\n📋 Test Case 3: Error Tracking")
    print("-" * 45)

    metadata.add_error("Network timeout during operation")
    metadata.tool_orchestration_error = "Failed to generate custom tool"
    metadata.web_search_error = "Search API rate limit exceeded"

    print(f"✅ Error tracking:")
    print(f"   - General errors: {metadata.errors}")
    print(f"   - Tool error: {metadata.tool_orchestration_error}")
    print(f"   - Search error: {metadata.web_search_error}")
    print(f"   - Has errors: {metadata.has_errors()}")

    print(f"\n📋 Test Case 4: Integration with WorkflowState")
    print("-" * 45)

    # Create WorkflowState with strongly typed metadata
    state = WorkflowState(
        messages=[LangChainMessage(content="Test message", type="user")],
        user_id="test_user",
        execution_metadata=metadata,
    )

    # Demonstrate type-safe access
    print(f"✅ WorkflowState integration:")
    print(f"   - Metadata type: {type(state.execution_metadata).__name__}")
    print(f"   - Streaming enabled: {state.execution_metadata.streaming_enabled}")
    print(
        f"   - Tool orchestration success: {state.execution_metadata.orchestration_success}"
    )
    print(f"   - Search completed: {state.execution_metadata.search_completed}")

    print(f"\n📋 Test Case 5: Extensibility with Extra Fields")
    print("-" * 45)

    # Demonstrate extra field support (model_config allows extra fields)
    setattr(metadata, "custom_metric", "example_value")
    setattr(metadata, "performance_score", 0.95)

    print(f"✅ Extended metadata:")
    print(f"   - Custom metric: {getattr(metadata, 'custom_metric', 'Not set')}")
    print(
        f"   - Performance score: {getattr(metadata, 'performance_score', 'Not set')}"
    )

    print(f"\n🎉 Strongly Typed ExecutionMetadata Demo Complete!")

    print(f"\n📋 Key Benefits Demonstrated:")
    print("   1. ✅ Type Safety - Fields have proper types and validation")
    print("   2. ✅ API Usability - Structured methods for common operations")
    print("   3. ✅ Documentation - Self-documenting field descriptions")
    print("   4. ✅ Error Handling - Centralized error tracking methods")
    print("   5. ✅ IDE Support - Auto-completion and type checking")
    print("   6. ✅ Extensibility - Allows additional fields when needed")
    print("   7. ✅ Validation - Pydantic validation for data integrity")

    print(f"\n🔄 Migration Benefits:")
    print("   - Before: metadata['field_name'] # No type safety, typo-prone")
    print("   - After:  metadata.field_name    # Type-safe, IDE support")
    print("   - Before: if metadata.get('errors'): # Manual error checking")
    print("   - After:  if metadata.has_errors(): # Structured error API")


if __name__ == "__main__":
    demonstrate_strongly_typed_execution_metadata()
