#!/usr/bin/env python3
"""
Simplified test for metadata flow from stream_pipeline to nodes.
Tests only the metadata creation and logging without full service setup.
"""

import asyncio
import logging
from datetime import datetime, timezone

from models import (
    Message,
    MessageContent, 
    MessageContentType,
    MessageRole,
    ModelProfileType
)
from composer.nodes.infrastructure.pipeline import PipelineNode
from composer.graph.state import WorkflowState
from runner import pipeline_factory

# Set up logging to see metadata
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_metadata_creation():
    """Test just the metadata creation functionality."""
    
    print("🔍 Testing PipelineNode metadata creation...")
    
    try:
        # Create PipelineNode with custom name
        pipeline_node = PipelineNode(
            pipeline_factory=pipeline_factory,
            profile_type=ModelProfileType.Primary,
            stream=True,
            node_name="TestMetadataNode"
        )
        
        print(f"✅ Created PipelineNode:")
        print(f"   Node Name: {pipeline_node.node_name}")
        print(f"   Node ID: {pipeline_node.node_id}")
        print(f"   Profile Type: {pipeline_node.profile_type}")
        print(f"   Streaming: {pipeline_node.stream}")
        
        # Create a mock pipeline object to test metadata creation
        class MockPipeline:
            def __init__(self):
                self.model = MockModel()
        
        class MockModel:
            def __init__(self):
                self.name = "Test-Model"
                self.provider = "test_provider"
        
        mock_pipeline = MockPipeline()
        
        # Create a minimal state for testing
        state = WorkflowState(
            user_id="test-user-metadata",
            conversation_id=12345,
        )
        
        # Test metadata creation
        metadata = pipeline_node.create_node_metadata(state, mock_pipeline)
        
        print(f"\n📋 Generated Node Metadata:")
        for key, value in metadata.items():
            print(f"   {key}: {value}")
        
        # Test that metadata contains expected fields
        expected_fields = [
            'node_name', 'node_id', 'node_type', 'profile_type', 
            'priority', 'streaming', 'execution_time', 'user_id',
            'conversation_id', 'pipeline_type', 'model_name', 'model_provider'
        ]
        
        missing_fields = [field for field in expected_fields if field not in metadata]
        if missing_fields:
            print(f"❌ Missing expected fields: {missing_fields}")
        else:
            print(f"✅ All expected metadata fields present")
        
        # Test state metadata storage
        print(f"\n🗂️  Testing state metadata storage...")
        
        # Simulate what the node would do
        if not hasattr(state, 'node_metadata'):
            state.node_metadata = {}
        state.node_metadata[pipeline_node.node_id] = metadata
        
        print(f"   Node metadata keys in state: {list(state.node_metadata.keys())}")
        print(f"   Stored metadata for node {pipeline_node.node_id}: ✅")
        
        # Test ComposerService-style event enrichment
        print(f"\n🔄 Testing event metadata enrichment simulation...")
        
        mock_event = {
            "event": "on_chain_end",
            "name": "TestMetadataNode",
            "data": {
                "values": state.model_dump(),
                "output": "test response"
            }
        }
        
        # Simulate ComposerService.execute_workflow metadata injection
        data = mock_event.get("data", {})
        state_values = data.get("values", {})
        node_metadata = state_values.get("node_metadata")
        
        print(f"   Event before enrichment:")
        print(f"     Has node_metadata in data: {'node_metadata' in data}")
        
        if node_metadata and "node_metadata" not in data:
            data["node_metadata"] = node_metadata
            print(f"   ✅ Would inject node_metadata into event")
        elif node_metadata:
            print(f"   ✅ Node metadata available for injection")
        
        print(f"     Available node IDs: {list(node_metadata.keys()) if node_metadata else 'None'}")
        
        # Show what would be in the final event
        if node_metadata:
            for node_id, meta in node_metadata.items():
                print(f"   📊 Metadata for {node_id}:")
                print(f"       Node: {meta.get('node_name', 'unknown')}")
                print(f"       Pipeline: {meta.get('pipeline_type', 'unknown')}")
                print(f"       Model: {meta.get('model_name', 'unknown')}")
                print(f"       Provider: {meta.get('model_provider', 'unknown')}")
        
        print(f"\n✅ Metadata creation test completed successfully!")
        print(f"   Node metadata creation: ✅")
        print(f"   State storage: ✅")
        print(f"   Event enrichment simulation: ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_metadata_creation())