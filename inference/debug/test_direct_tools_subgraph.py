#!/usr/bin/env python3

"""Test direct tool execution through ToolsAgentSubgraph vs ChatAgent"""

import asyncio
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
from composer.graph.state import WorkflowState
from models.default_configs import create_default_user_config

async def test_direct_tools_subgraph():
    print("🧪 Testing direct ToolsAgentSubgraph execution...")
    
    # Create subgraph
    subgraph = ToolsAgentSubgraph()
    
    # Create minimal state with web search tool call
    state = WorkflowState(
        user_id='test_user',
        conversation_id=1,
        user_config=create_default_user_config('test_user'),
        messages=[],
        things_to_remember=[],
        # Simulate tool calls like the LLM would generate
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": "AI developments 2024"},
                "id": "test_call_1",
                "type": "tool_call"
            }
        ]
    )
    
    try:
        # Execute the subgraph
        print("🚀 Executing ToolsAgentSubgraph with direct tool call...")
        result = await subgraph.graph.ainvoke(state)
        print(f"✅ ToolsAgentSubgraph execution successful: {len(result.get('messages', []))} messages returned")
        return True
    except Exception as e:
        print(f"❌ ToolsAgentSubgraph execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(test_direct_tools_subgraph())