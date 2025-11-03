#!/usr/bin/env python3
"""
Test real composer workflow execution to trace user_id through the complete flow.

This test simulates the actual composer workflow execution to see where user_id
gets lost between initial state creation and memory tool execution.
"""

import asyncio
from datetime import datetime, timezone
from models import LangChainMessage
from composer.graph.state import WorkflowState
from composer.core.service import ComposerService
from utils.logging import llmmllogger

logger = llmmllogger.logger.bind(component="WorkflowUserIdTrace")


async def test_real_composer_workflow_user_id():
    """Test real composer workflow to trace user_id through execution."""
    
    print("=" * 70)
    print("🧪 TESTING REAL COMPOSER WORKFLOW USER_ID TRACE") 
    print("=" * 70)
    
    # Create composer service
    composer_service = ComposerService()
    test_user_id = "test-user-trace-123"
    test_conversation_id = 9999
    
    try:
        # Step 1: Create initial state (same as real workflow)
        print(f"\n📝 Step 1: Creating initial state for user_id='{test_user_id}'")
        initial_state = await composer_service.create_initial_state(
            user_id=test_user_id,
            conversation_id=test_conversation_id,
        )
        
        print(f"   ✅ Initial state created")
        print(f"   - user_id: '{initial_state.user_id}'")
        print(f"   - Type: {type(initial_state.user_id)}")
        print(f"   - Is truthy: {bool(initial_state.user_id)}")
        print(f"   - conversation_id: {initial_state.conversation_id}")
        print(f"   - messages count: {len(initial_state.messages) if initial_state.messages else 0}")
        print(f"   - user_config present: {initial_state.user_config is not None}")
        
        if not initial_state.user_id:
            print(f"   ❌ CRITICAL: Initial state has empty user_id!")
            return False
            
        # Step 2: Try to build workflow (this loads configuration)
        print(f"\n🔧 Step 2: Building workflow graph for user_id='{test_user_id}'")
        workflow = await composer_service.compose_workflow(test_user_id)
        print(f"   ✅ Workflow built successfully")
        
        # Step 3: Check if the state still has user_id after workflow creation
        print(f"\n🔍 Step 3: Checking state after workflow creation")
        print(f"   - user_id still present: '{initial_state.user_id}'")
        print(f"   - user_id is truthy: {bool(initial_state.user_id)}")
        
        # Step 4: Simulate transform_to_tools_state
        print(f"\n🔄 Step 4: Simulating tools agent transformation")
        from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
        from composer.tools.registry import ToolRegistry
        from composer.agents.chat_agent import ChatAgent
        
        # Create minimal components for ToolsAgentSubgraph (just to test transformation)
        tool_registry = ToolRegistry()
        
        # We need a ChatAgent instance - let's check if we can create a minimal one
        try:
            from models.default_configs import DEFAULT_USER_CONFIG  
            from db import storage
            
            # Try to get user config from storage
            user_config = await storage.get_service(storage.user_config).get_user_config(test_user_id)
            if not user_config:
                print(f"   ⚠️ No user config found for user_id='{test_user_id}', using defaults")
                user_config = DEFAULT_USER_CONFIG._replace(user_id=test_user_id)
            
            chat_agent = ChatAgent(user_config=user_config)
            tools_agent = ToolsAgentSubgraph(tool_registry, chat_agent)
            
            # Now test the transformation
            tools_state = tools_agent.transform_to_tools_state(initial_state)
            
            print(f"   ✅ Transformation completed")
            print(f"   - ToolsState user_id: '{tools_state.get('user_id', 'NOT_FOUND')}'")
            print(f"   - ToolsState user_id type: {type(tools_state.get('user_id', 'NOT_FOUND'))}")
            print(f"   - ToolsState user_id is truthy: {bool(tools_state.get('user_id', ''))}")
            print(f"   - conversation_id: {tools_state.get('conversation_id', 'NOT_FOUND')}")
            
            # Step 5: Test memory tool state check
            print(f"\n🛠️ Step 5: Testing memory tool state check")
            user_id_from_tools_state = tools_state.get("user_id")
            print(f"   - tools_state.get('user_id'): '{user_id_from_tools_state}'")
            print(f"   - not tools_state.get('user_id'): {not tools_state.get('user_id')}")
            
            if not tools_state.get("user_id"):
                print(f"   ❌ MEMORY TOOL WOULD FAIL: Missing user_id in ToolsState")
                return False
            else:
                print(f"   ✅ MEMORY TOOL WOULD PASS: user_id found in ToolsState")
                return True
                
        except Exception as e:
            print(f"   ❌ Failed to create ToolsAgentSubgraph: {e}")
            print(f"   🔍 This might be the source of the issue in production")
            return False
            
    except Exception as e:
        logger.error(f"❌ Workflow test failed: {e}", exc_info=True)
        print(f"   ❌ Test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🏃 Starting real composer workflow user_id trace...")
    
    success = await test_real_composer_workflow_user_id()
    
    print(f"\n" + "=" * 70)
    if success:
        print("✅ WORKFLOW USER_ID TRACE TEST PASSED")
        print("   user_id correctly flows through entire workflow creation chain")
    else:
        print("❌ WORKFLOW USER_ID TRACE TEST FAILED") 
        print("   user_id was lost somewhere in the workflow creation chain")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())