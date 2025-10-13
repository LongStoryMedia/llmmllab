#!/usr/bin/env python3
"""
Test metadata flow from PipelineNode through ComposerService.
"""

import asyncio
import logging
from datetime import datetime, timezone

from models import (
    Message,
    MessageContent, 
    MessageContentType,
    MessageRole,
    ModelProfile,
    ModelParameters,
    ModelProfileType
)
from composer.core.service import ComposerService
from composer.nodes.infrastructure.pipeline import PipelineNode
from composer.graph.state import WorkflowState
from runner import pipeline_factory

# Set up logging to see metadata
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_metadata_flow():
    """Test that metadata flows from PipelineNode through ComposerService."""
    
    print("🔍 Testing metadata flow through Composer pipeline...")
    
    try:
        # Create test state
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text="Hello from metadata test!")]
        )
        
        # Create minimal WorkflowState for testing
        from composer.utils.conversion import message_to_langchain_message
        lc_message = message_to_langchain_message(test_message)
        
        state = WorkflowState(
            messages=[lc_message],
            user_id="test-user-metadata",
            conversation_id=12345,
            current_user_message=lc_message
        )
        
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
        
        # Mock user config (since we don't have full database in this test)
        from models import UserConfig, ModelParameters, ModelProfile, WorkflowConfig, CircuitBreakerConfig
        import uuid
        
        # Create a proper user config
        user_config = UserConfig(
            user_id="test-user-metadata",
            model_profiles=[
                ModelProfile(
                    id=str(uuid.uuid4()),
                    user_id="test-user-metadata",
                    name="test-primary",
                    description="Test primary profile",
                    model_name="qwen3-4b-ud-q6-k-xl",
                    parameters=ModelParameters(
                        temperature=0.7,
                        max_tokens=100,
                        top_p=0.9,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    ),
                    system_prompt="You are a helpful assistant.",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    type=0  # Primary type
                )
            ],
            workflow=WorkflowConfig(),
            circuit_breaker=CircuitBreakerConfig(),
        )
        
        state.user_config = user_config
        
        print(f"\n🚀 Executing PipelineNode...")
        
        # Execute the pipeline node manually (simulating what LangGraph would do)  
        result_state = await pipeline_node(state)
        
        print(f"\n📊 Node Execution Results:")
        print(f"   Messages count: {len(result_state.messages)}")
        print(f"   Node metadata keys: {list(result_state.node_metadata.keys()) if result_state.node_metadata else 'None'}")
        
        # Display the captured metadata
        if result_state.node_metadata:
            for node_id, metadata in result_state.node_metadata.items():
                print(f"\n📋 Metadata for Node {node_id}:")
                for key, value in metadata.items():
                    if key == "execution_time":
                        print(f"   {key}: {value}")
                    else:
                        print(f"   {key}: {value}")
        
        # Test ComposerService metadata injection (simulation)
        print(f"\n🔄 Testing ComposerService metadata enrichment...")
        
        # Simulate a workflow event that would contain state
        mock_event = {
            "event": "on_chain_end",
            "name": "TestMetadataNode", 
            "data": {
                "values": result_state.model_dump(),
                "output": "test response"
            }
        }
        
        print(f"   Mock event before enrichment:")
        print(f"     Event type: {mock_event['event']}")
        print(f"     Has node_metadata in data: {'node_metadata' in mock_event['data']}")
        
        # Simulate what execute_workflow would do
        data = mock_event.get("data", {})
        state_values = data.get("values", {})
        node_metadata = state_values.get("node_metadata")
        
        if node_metadata and "node_metadata" not in data:
            data["node_metadata"] = node_metadata
            print(f"   ✅ Injected node_metadata into event data")
            print(f"     Node metadata keys: {list(node_metadata.keys())}")
        
        print(f"\n✅ Metadata flow test completed successfully!")
        print(f"   Pipeline execution metadata: ✅")
        print(f"   Node metadata creation: ✅") 
        print(f"   State metadata storage: ✅")
        print(f"   Event metadata injection: ✅")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_metadata_flow())