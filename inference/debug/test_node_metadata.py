"""
Test script to verify NodeMetadata integration with BaseNode.
"""

from datetime import datetime, timezone
from composer.nodes.base_node import BaseNode
from composer.graph.state import WorkflowState
from models.node_metadata import NodeMetadata, ErrorDetails


class TestNode(BaseNode):
    """Test node to verify BaseNode metadata functionality."""

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        return state


def test_node_metadata_creation():
    """Test that NodeMetadata objects are created correctly."""
    
    # Create test state
    state = WorkflowState()
    state.user_id = "test_user_123"
    state.conversation_id = 456
    
    # Create test node
    node = TestNode("TestNode")
    
    print("🧪 Testing NodeMetadata creation...")
    
    # Test basic metadata creation
    metadata = node.create_node_metadata(state)
    
    print(f"✅ Created NodeMetadata object: {type(metadata)}")
    print(f"   - node_name: {metadata.node_name}")
    print(f"   - node_id: {metadata.node_id}")
    print(f"   - node_type: {metadata.node_type}")
    print(f"   - execution_time: {metadata.execution_time}")
    print(f"   - user_id: {metadata.user_id}")
    print(f"   - conversation_id: {metadata.conversation_id}")
    
    # Test metadata with additional fields
    metadata_with_extras = node.create_node_metadata(
        state,
        model_name="test_model",
        profile_type="fast",
        streaming=True,
        is_cached=False,
        tool_count=3
    )
    
    print(f"✅ Created NodeMetadata with extras:")
    print(f"   - model_name: {metadata_with_extras.model_name}")
    print(f"   - profile_type: {metadata_with_extras.profile_type}")
    print(f"   - streaming: {metadata_with_extras.streaming}")
    print(f"   - is_cached: {metadata_with_extras.is_cached}")
    print(f"   - tool_count: {metadata_with_extras.tool_count}")
    
    # Test storage in state
    node.store_node_metadata(state, model_name="stored_model")
    
    print(f"✅ Stored metadata in state")
    print(f"   - state.node_metadata keys: {list(state.node_metadata.keys())}")
    print(f"   - stored metadata type: {type(state.node_metadata[node.node_id])}")
    
    # Test error metadata
    error_metadata = node.create_node_metadata(
        state,
        error_details=ErrorDetails(
            error_type="TestError",
            error_message="This is a test error",
            stack_trace="line 1\nline 2\nline 3"
        )
    )
    
    print(f"✅ Created NodeMetadata with error details:")
    print(f"   - error_type: {error_metadata.error_details.error_type}")
    print(f"   - error_message: {error_metadata.error_details.error_message}")
    
    # Test JSON serialization
    metadata_dict = metadata.model_dump()
    print(f"✅ Serialized to dict: {len(metadata_dict)} fields")
    
    print("🎉 All NodeMetadata tests passed!")


if __name__ == "__main__":
    test_node_metadata_creation()