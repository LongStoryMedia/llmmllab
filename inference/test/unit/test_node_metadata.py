"""
Unit tests for NodeMetadata integration with BaseNode.
"""

import pytest
from datetime import datetime, timezone
from composer.nodes.base_node import BaseNode
from composer.graph.state import WorkflowState
from models.node_metadata import NodeMetadata, ErrorDetails


class TestNodeForMetadata(BaseNode):
    """Test node to verify BaseNode metadata functionality."""

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        return state


def create_test_state() -> WorkflowState:
    """Create test WorkflowState."""
    state = WorkflowState()
    state.user_id = "test_user_123"
    state.conversation_id = 456
    return state


def create_test_node() -> TestNodeForMetadata:
    """Create test node."""
    return TestNodeForMetadata("TestNode")


def test_basic_metadata_creation():
    """Test basic NodeMetadata object creation."""
    state = create_test_state()
    node = create_test_node()
    metadata = node.create_node_metadata(state)

    assert isinstance(metadata, NodeMetadata)
    assert metadata.node_name == "TestNode"
    assert metadata.node_id == node.node_id
    assert metadata.node_type == "TestNodeForMetadata"
    assert isinstance(metadata.execution_time, datetime)
    assert metadata.user_id == "test_user_123"
    assert metadata.conversation_id == 456


def test_metadata_with_optional_fields():
    """Test NodeMetadata creation with optional fields."""
    state = create_test_state()
    node = create_test_node()
    metadata = node.create_node_metadata(
        state,
        model_name="test_model",
        profile_type="fast",
        streaming=True,
        is_cached=False,
        tool_count=3,
        intent_classification="chat",
        memory_operations=["store", "retrieve"],
    )

    assert metadata.profile_type == "fast"
    assert metadata.streaming is True
    assert metadata.is_cached is False
    assert metadata.tool_count == 3


def test_store_node_metadata():
    """Test storing metadata in workflow state."""
    state = create_test_state()
    node = create_test_node()
    node.store_node_metadata(state, model_name="stored_model")

    assert hasattr(state, "node_metadata")
    assert node.node_id in state.node_metadata

    stored_metadata = state.node_metadata[node.node_id]
    assert isinstance(stored_metadata, NodeMetadata)


def test_metadata_serialization():
    """Test NodeMetadata JSON serialization."""
    state = create_test_state()
    node = create_test_node()
    metadata = node.create_node_metadata(state, model_name="test_model", streaming=True)

    metadata_dict = metadata.model_dump()

    assert isinstance(metadata_dict, dict)
    assert metadata_dict["node_name"] == "TestNode"
    assert metadata_dict["model_name"] == "test_model"
    assert metadata_dict["streaming"] is True
    assert "execution_time" in metadata_dict


def test_enrich_with_node_metadata():
    """Test enriching objects with node metadata."""
    state = create_test_state()
    node = create_test_node()

    class TestObject:
        def __init__(self) -> None:
            self.data = "test"

    obj = TestObject()
    enriched_obj = node.enrich_with_node_metadata(
        obj, state, model_name="enrichment_model"
    )

    assert hasattr(enriched_obj, "node_metadata")
    node_metadata = getattr(enriched_obj, "node_metadata")
    assert isinstance(node_metadata, NodeMetadata)


def test_metadata_validation():
    """Test Pydantic validation of NodeMetadata fields."""
    # Test valid metadata creation
    metadata = NodeMetadata(
        node_name="ValidNode",
        node_id="valid_id",
        node_type="ValidType",
        execution_time=datetime.now(timezone.utc),
    )
    assert metadata.node_name == "ValidNode"

    # Test validation failure for missing required fields
    with pytest.raises((ValueError, TypeError)):
        # This will raise a TypeError for missing required fields
        NodeMetadata(node_name="InvalidNode")  # type: ignore


def test_error_details_validation():
    """Test ErrorDetails validation."""
    error_details = ErrorDetails(
        error_type="ValidationError", error_message="Field validation failed"
    )

    assert error_details.error_type == "ValidationError"
    assert error_details.error_message == "Field validation failed"
    assert error_details.stack_trace is None  # Optional field


def test_metadata_with_none_values():
    """Test NodeMetadata handles None values correctly for optional fields."""
    node = create_test_node()
    state_without_ids = WorkflowState()
    # No user_id or conversation_id set

    metadata = node.create_node_metadata(state_without_ids)

    assert metadata.user_id is None
    assert metadata.conversation_id is None


def test_multiple_metadata_storage():
    """Test storing metadata from multiple nodes."""
    state = create_test_state()
    node1 = TestNodeForMetadata("Node1")
    node2 = TestNodeForMetadata("Node2")

    node1.store_node_metadata(state, model_name="model1")
    node2.store_node_metadata(state, model_name="model2")

    assert len(state.node_metadata) == 2
    assert node1.node_id in state.node_metadata
    assert node2.node_id in state.node_metadata
