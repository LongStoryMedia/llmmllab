#!/usr/bin/env python3
"""
Isolated Engineering Agent Test

This test directly calls the engineering agent's generate_technical_response method
to verify what prompt and query it receives, and what response it generates.
This bypasses the workflow entirely to isolate the engineering agent behavior.
"""

import asyncio
import sys
import os

# Add the inference directory to the Python path
sys.path.insert(0, '/app')

async def test_engineering_agent_isolated():
    """Test engineering agent by directly calling the method used by engineering node."""
    print("� ISOLATED ENGINEERING AGENT TEST")
    print("="*60)
    
    try:
        # Import after path setup
        from composer import create_initial_state, compose_workflow
        from models import TechnicalDomain, ResponseFormat
        
        print("📋 Creating test state...")
        
        # Create initial state for engineering query
        test_query = "Help me create a FastAPI application with authentication"
        user_id = "test_user_isolated"
        
        # Create minimal state for testing
        initial_state = await create_initial_state(
            user_message=test_query,
            user_id=user_id,
            conversation_id=9999,  # Use high number to avoid conflicts
        )
        
        print(f"📝 Test Query: '{test_query}'")
        print(f"👤 User ID: {user_id}")
        print()
        
        # Get workflow components
        workflow = await compose_workflow(initial_state.user_config)
        
        # Extract engineering agent from workflow
        if hasattr(workflow, 'graph_builder') and hasattr(workflow.graph_builder, 'engineering_agent'):
            engineering_agent = workflow.graph_builder.engineering_agent
            
            print("� Testing engineering agent directly...")
            print("-" * 60)
            
            # Test the exact method call used by the engineering node
            response = await engineering_agent.generate_technical_response(
                query=test_query,
                user_id=user_id,
                domain=TechnicalDomain.SOFTWARE_DEVELOPMENT,
                response_format=ResponseFormat.CODE_SOLUTION
            )
            
            print("📄 ENGINEERING AGENT RESPONSE:")
            print(response)
            print("-" * 60)
            
            # Analyze the response
            if "Hello! I'm here to help with engineering tasks" in response:
                print("❌ CONFIRMED: Engineering agent gives generic greeting")
                print("   The agent receives the correct query but model responds generically")
                print("   Issue is in model behavior, not query passing")
            elif "FastAPI" in response and "authentication" in response:
                print("✅ SUCCESS: Engineering agent gives specific technical response")
            else:
                print("⚠️  UNCLEAR: Response doesn't match expected patterns")
        else:
            print("❌ Could not access engineering agent from workflow")
            
    except Exception as e:
        print(f"❌ Error in isolated test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_engineering_agent_isolated())