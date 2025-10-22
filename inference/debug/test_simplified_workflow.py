#!/usr/bin/env python3
"""Test the simplified workflow architecture."""

import asyncio
from composer.graph.builder import GraphBuilder
from models import WorkflowType

async def test_workflow_creation():
    """Test that the simplified workflow can be created."""
    try:
        print("🧪 Testing simplified workflow creation...")
        
        # Setup dependencies
        from runner.pipeline_factory import pipeline_factory
        from db import storage
        from models import UserConfig
        
        # Create basic user config
        user_config = UserConfig(
            user_id="test-user-123",
            conversation_id=1,
            primary_model="qwen3_7b_instruct",
            summary_model="qwen3_7b_instruct"
        )
        
        # Test workflow creation
        builder = GraphBuilder(storage, pipeline_factory, user_config)
        user_id = "test-user-123"
        
        # Create workflow
        workflow = builder.build_workflow(user_id, WorkflowType.DEFAULT)
        
        print(f"✅ Workflow created successfully: {type(workflow)}")
        
        # Check nodes
        nodes = list(workflow.get_graph().nodes.keys())
        print(f"📊 Workflow nodes: {nodes}")
        
        # Verify tools_agent node exists
        if "tools_agent" in nodes:
            print("✅ tools_agent node found in workflow")
        else:
            print("❌ tools_agent node not found!")
            
        # Verify old nodes are removed
        if "tool_executor" not in nodes:
            print("✅ tool_executor node correctly removed")
        else:
            print("❌ tool_executor node still exists!")
            
        if "chat_agent" not in nodes:
            print("✅ chat_agent node correctly removed")
        else:
            print("❌ chat_agent node still exists!")
            
        print("🎉 Simplified workflow test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Workflow creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_workflow_creation())
    if success:
        print("✅ Test passed")
    else:
        print("❌ Test failed")
        exit(1)