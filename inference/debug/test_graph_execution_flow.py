#!/usr/bin/env python3
"""
Debug the main graph execution flow to understand duplicate chat_agent executions.
"""

def analyze_graph_structure():
    """Analyze the graph structure to understand routing."""
    
    print("🔍 Graph Structure Analysis")
    print("=" * 60)
    
    print("📊 Expected routing logic:")
    print("")
    print("🔄 Main Graph Nodes:")
    print("   • memory_search → static_tool_loading → intent_analysis")
    print("   • intent_analysis → tool_collection → tool_composer → workflow_router")
    print("   • workflow_router → chat_agent (primary routing)")
    print("   • chat_agent → should_execute_tools()")
    print("     ├─ has_tool_calls=True → tool_executor")
    print("     └─ has_tool_calls=False → chat_summary")
    print("   • tool_executor → should_continue_agent_loop()")
    print("     ├─ web_search_results → search_summary → memory_creation")
    print("     └─ no_web_search → memory_creation")
    print("   • memory_creation → memory_storage → END")
    print("")
    print("🔄 Subgraph (tools_agent):")
    print("   • START → chat_agent → tools_condition()")
    print("     ├─ has_tool_calls=True → tool_executor → END")
    print("     └─ has_tool_calls=False → END")
    print("")
    print("❌ Problem Analysis:")
    print("   1. Main chat_agent generates tool calls")
    print("   2. Routes to tool_executor (✅ correct)")
    print("   3. tool_executor calls subgraph")
    print("   4. Subgraph executes and ends (✅ fixed)")
    print("   5. tool_executor should route to memory_creation")
    print("   6. **BUT main chat_agent is called again!**")
    print("")
    print("🤔 Possible Causes:")
    print("   • State modification in subgraph triggers re-routing")
    print("   • Message added by subgraph has tool calls")
    print("   • LangGraph execution order issue")
    print("   • tool_executor not properly finishing")


def analyze_log_pattern():
    """Analyze the log pattern from the previous test."""
    
    print("\n🔍 Log Pattern Analysis")
    print("=" * 60)
    
    print("📊 Observed Execution Pattern:")
    print("")
    print("🕐 Timeline Analysis:")
    print("   16:16:22 - PrimaryChatAgent call #1 (generates 3 tool calls)")
    print("   16:16:22 - Routes to tool_executor")
    print("   16:16:22 - subgraph_chat_agent executes (no tool calls)")
    print("   16:16:24 - PrimaryChatAgent call #2 (DUPLICATE! same 3 tool calls)")
    print("   16:17:19 - PrimaryChatAgent call #3 (DUPLICATE! same 3 tool calls)")
    print("")
    print("🔍 Key Observations:")
    print("   • Same tool calls generated each time:")
    print("     - web_search: 'Major AI model releases 2024'")
    print("     - web_search: 'Recent AI research breakthroughs 2024'")
    print("     - web_search: 'AI safety developments 2024'")
    print("   • Each execution routes to tool_executor correctly")
    print("   • subgraph_chat_agent runs but produces no tool calls")
    print("   • Message count increases (2→4 messages)")
    print("")
    print("💡 Root Cause Hypothesis:")
    print("   The subgraph is NOT ending properly after tool execution.")
    print("   Even though we fixed the routing to END, something is")
    print("   causing the main graph to re-execute chat_agent.")
    print("")
    print("🔧 Investigation Steps:")
    print("   1. Check if ToolExecutorNode properly handles subgraph result")
    print("   2. Verify subgraph actually reaches END state")
    print("   3. Check if tool_executor routes to memory_creation")
    print("   4. Look for any loops in main graph structure")


def examine_tool_executor_node():
    """Examine the ToolExecutorNode logic."""
    
    print("\n🔍 ToolExecutorNode Analysis")
    print("=" * 60)
    
    print("📊 ToolExecutorNode Logic:")
    print("   1. Receives WorkflowState from main graph")
    print("   2. Calls tools_agent_subgraph.execute(state)")
    print("   3. Gets Command result from subgraph")
    print("   4. Applies command.update to state")
    print("   5. Returns updated state to main graph")
    print("")
    print("🤔 Potential Issues:")
    print("   • Subgraph might return messages with tool_calls")
    print("   • State update might trigger should_execute_tools again")
    print("   • tool_executor might not route to memory_creation")
    print("")
    print("🔧 Fix Strategy:")
    print("   1. Ensure subgraph only returns tool results, not new tool calls")
    print("   2. Verify should_continue_agent_loop routes correctly")
    print("   3. Add logging to track state transitions")
    print("   4. Consider bypassing subgraph entirely")


async def test_simple_execution():
    """Simplified test without complex imports."""
    
    print("\n🔍 Simplified Execution Test")
    print("=" * 60)
    
    print("🧪 Testing Theory:")
    print("   The issue is likely that after tool_executor completes,")
    print("   the state contains messages that trigger should_execute_tools")
    print("   to route back to chat_agent instead of memory_creation.")
    print("")
    print("📊 Expected vs Actual:")
    print("   Expected: chat_agent → tool_executor → memory_creation")
    print("   Actual:   chat_agent → tool_executor → chat_agent (loop!)")
    print("")
    print("🎯 Solution:")
    print("   Modify should_execute_tools to only check the LAST message")
    print("   from the PRIMARY chat_agent, not from subgraph results.")
    
    return True


def test_routing_hypothesis():
    """Test our hypothesis about the routing issue."""
    
    print("\n🔍 Routing Hypothesis Test")
    print("=" * 60)
    
    print("💡 HYPOTHESIS:")
    print("   The issue is in should_execute_tools routing logic.")
    print("   After tool_executor completes, new messages are added")
    print("   that cause should_execute_tools to route back to chat_agent.")
    print("")
    print("� Evidence from logs:")
    print("   • Message count grows: 2 → 4 messages")
    print("   • should_execute_tools shows 'Last message type=ai, has_tool_calls=True'")
    print("   • This triggers routing back to tool_executor again")
    print("")
    print("🔧 Solution Strategy:")
    print("   Modify should_execute_tools to be more selective:")
    print("   1. Only process tool calls from PRIMARY chat agent")
    print("   2. Ignore tool calls from subgraph/tool results")
    print("   3. Track message sources to prevent loops")
    print("")
    print("🎯 Key Insight:")
    print("   The subgraph might be adding an AI message with tool_calls")
    print("   that gets interpreted as 'new work to do' by the main graph.")
    
    return True


def analyze_graph_structure():
    """Analyze the graph structure to understand routing."""
    
    print("\n\n🔍 Graph Structure Analysis")
    print("=" * 60)
    
    print("📊 Expected routing logic:")
    print("")
    print("🔄 Main Graph Nodes:")
    print("   • memory_search → static_tool_loading → intent_analysis")
    print("   • intent_analysis → tool_collection → tool_composer → workflow_router")
    print("   • workflow_router → chat_agent (primary routing)")
    print("   • chat_agent → should_execute_tools()")
    print("     ├─ has_tool_calls=True → tool_executor")
    print("     └─ has_tool_calls=False → chat_summary")
    print("   • tool_executor → should_continue_agent_loop()")
    print("     ├─ web_search_results → search_summary → memory_creation")
    print("     └─ no_web_search → memory_creation")
    print("   • memory_creation → memory_storage → END")
    print("")
    print("🔄 Subgraph (tools_agent):")
    print("   • START → chat_agent → tools_condition()")
    print("     ├─ has_tool_calls=True → tool_executor → END")
    print("     └─ has_tool_calls=False → END")
    print("")
    print("❌ Problem Analysis:")
    print("   1. Main chat_agent generates tool calls")
    print("   2. Routes to tool_executor (✅ correct)")
    print("   3. tool_executor calls subgraph")
    print("   4. Subgraph executes and ends (✅ fixed)")
    print("   5. tool_executor should route to memory_creation")
    print("   6. **BUT main chat_agent is called again!**")
    print("")
    print("🤔 Possible Causes:")
    print("   • State modification in subgraph triggers re-routing")
    print("   • Message added by subgraph has tool calls")
    print("   • LangGraph execution order issue")
    print("   • tool_executor not properly finishing")


if __name__ == "__main__":
    print("🔍 Graph Execution Flow Debug")
    print("=" * 60)
    
    # Run the analysis
    analyze_graph_structure()
    analyze_log_pattern()
    examine_tool_executor_node()
    test_routing_hypothesis()
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("   1. Check if subgraph returns messages with tool_calls")
    print("   2. Verify tool_executor routing logic")
    print("   3. Modify should_execute_tools to prevent loops")
    print("   4. Add message source tracking to distinguish origins")