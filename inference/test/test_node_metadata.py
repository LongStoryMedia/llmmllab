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


class TestNodeMetadata:
    """Unit tests for NodeMetadata functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.state = WorkflowState()
        self.state.user_id = "test_user_123"
        self.state.conversation_id = 456
        self.node = TestNodeForMetadata("TestNode")

    def test_basic_metadata_creation(self):
        """Test basic NodeMetadata object creation."""
        metadata = self.node.create_node_metadata(self.state)
        
        assert isinstance(metadata, NodeMetadata)
        assert metadata.node_name == "TestNode"
        assert metadata.node_id == self.node.node_id
        assert metadata.node_type == "TestNodeForMetadata"
        assert isinstance(metadata.execution_time, datetime)
        assert metadata.user_id == "test_user_123"
        assert metadata.conversation_id == 456

    def test_metadata_with_optional_fields(self):
        """Test NodeMetadata creation with optional fields."""
        metadata = self.node.create_node_metadata(
            self.state,
            model_name="test_model",
            profile_type="fast", 
            streaming=True,
            is_cached=False,
            tool_count=3,
            intent_classification="chat",
            memory_operations=["store", "retrieve"]
        )
        
        assert metadata.model_name == "test_model"
        assert metadata.profile_type == "fast"
        assert metadata.streaming is True
        assert metadata.is_cached is False
        assert metadata.tool_count == 3
        assert metadata.intent_classification == "chat"
        assert metadata.memory_operations == ["store", "retrieve"]

    def test_metadata_with_error_details(self):
        """Test NodeMetadata creation with error information."""
        error_details = ErrorDetails(
            error_type="TestError",
            error_message="This is a test error",
            stack_trace="line 1\nline 2\nline 3"
        )
        
        metadata = self.node.create_node_metadata(
            self.state,
            error_details=error_details
        )
        
        assert metadata.error_details is not None
        assert metadata.error_details.error_type == "TestError"
        assert metadata.error_details.error_message == "This is a test error"
        assert metadata.error_details.stack_trace == "line 1\nline 2\nline 3"

    def test_store_node_metadata(self):
        """Test storing metadata in workflow state."""
        self.node.store_node_metadata(self.state, model_name="stored_model")
        
        assert hasattr(self.state, "node_metadata")
        assert self.node.node_id in self.state.node_metadata
        
        stored_metadata = self.state.node_metadata[self.node.node_id]
        assert isinstance(stored_metadata, NodeMetadata)
        assert stored_metadata.model_name == "stored_model"

    def test_metadata_serialization(self):
        """Test NodeMetadata JSON serialization."""
        metadata = self.node.create_node_metadata(
            self.state,
            model_name="test_model",
            streaming=True
        )
        
        metadata_dict = metadata.model_dump()
        
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["node_name"] == "TestNode"
        assert metadata_dict["model_name"] == "test_model"
        assert metadata_dict["streaming"] is True
        assert "execution_time" in metadata_dict

    def test_enrich_with_node_metadata(self):
        """Test enriching objects with node metadata."""
        class TestObject:
            def __init__(self):
                self.data = "test"
        
        obj = TestObject()
        enriched_obj = self.node.enrich_with_node_metadata(
            obj, 
            self.state, 
            model_name="enrichment_model"
        )
        
        assert hasattr(enriched_obj, "node_metadata")
        assert isinstance(enriched_obj.node_metadata, NodeMetadata)
        assert enriched_obj.node_metadata.model_name == "enrichment_model"

    def test_metadata_validation(self):
        """Test Pydantic validation of NodeMetadata fields."""
        # Test valid metadata creation
        metadata = NodeMetadata(
            node_name="ValidNode",
            node_id="valid_id",
            node_type="ValidType",
            execution_time=datetime.now(timezone.utc)
        )
        assert metadata.node_name == "ValidNode"
        
        # Test validation failure for missing required fields
        with pytest.raises(ValueError):
            NodeMetadata(
                node_name="InvalidNode"
                # Missing required fields should raise validation error
            )

    def test_error_details_validation(self):
        """Test ErrorDetails validation."""
        error_details = ErrorDetails(
            error_type="ValidationError",
            error_message="Field validation failed"
        )
        
        assert error_details.error_type == "ValidationError"
        assert error_details.error_message == "Field validation failed"
        assert error_details.stack_trace is None  # Optional field

    def test_metadata_with_none_values(self):
        """Test NodeMetadata handles None values correctly for optional fields."""
        state_without_ids = WorkflowState()
        # No user_id or conversation_id set
        
        metadata = self.node.create_node_metadata(state_without_ids)
        
        assert metadata.user_id is None
        assert metadata.conversation_id is None
        assert metadata.model_name is None  # Optional field defaults to None

    def test_multiple_metadata_storage(self):
        """Test storing metadata from multiple nodes."""
        node1 = TestNodeForMetadata("Node1")
        node2 = TestNodeForMetadata("Node2")
        
        node1.store_node_metadata(self.state, model_name="model1")
        node2.store_node_metadata(self.state, model_name="model2")
        
        assert len(self.state.node_metadata) == 2
        assert node1.node_id in self.state.node_metadata
        assert node2.node_id in self.state.node_metadata
        
        assert self.state.node_metadata[node1.node_id].model_name == "model1"
        assert self.state.node_metadata[node2.node_id].model_name == "model2"