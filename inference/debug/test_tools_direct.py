#!/usr/bin/env python3
"""
Direct Tools Agent Test - bypasses intent classification to test tools subgraph directly.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, Any

from composer.core.service import ComposerService
from db import storage
from models import (
    LangChainMessage,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
    WorkflowState,
)
from composer.graph.states import ToolsState
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
from composer.nodes.agents.tool_registry import ToolRegistry
from composer.agents.chat_agent import ChatAgent
from runner import PipelineFactory
from models.config import Config
import llmmllogger

async def test_tools_agent_direct():
    """Test tools agent subgraph directly to check response generation."""
    
    # Initialize storage
    await storage.init_db()
    config = Config.default()
    pipeline_factory = PipelineFactory(config)
    tool_registry = ToolRegistry()
    await tool_registry.initialize()
    
    # Load executable tools
    from composer.tools.web_search_tool import WebSearchTool
    from composer.tools.summarization_tool import SummarizationTool
    from composer.tools.memory_retrieval_tool import MemoryRetrievalTool
    from composer.tools.date_tool import GetCurrentDateTool
    
    executable_tools = {
        "web_search": WebSearchTool(),
        "summarization": SummarizationTool(),
        "memory_retrieval": MemoryRetrievalTool(),
        "get_current_date": GetCurrentDateTool(),
    }
    tool_registry.executable_tools = executable_tools
    
    logger = llmmllogger.logger.bind(component="direct_tools_test")
    logger.info("🧪 Testing Tools Agent Direct Execution")
    
    # Create ChatAgent with medium model 
    from models import ModelProfile, NodeMetadata, PipelinePriority
    model_profile = ModelProfile.get_model("qwen3-30b-a3b-q4-k-m")
    node_metadata = NodeMetadata(
        node_name="TestChatAgent", 
        node_id="test_agent",
        node_type="test",
        user_id="test_user",
        conversation_id=999
    )
    
    chat_agent = ChatAgent(
        pipeline_factory=pipeline_factory,
        profile=model_profile,
        node_metadata=node_metadata,
        priority=PipelinePriority.MEDIUM
    )
    
    # Create subgraph
    subgraph = ToolsAgentSubgraph(tool_registry, chat_agent)
    
    # Create initial state with human message
    from langchain_core.messages import HumanMessage
    
    initial_state = ToolsState(messages=[
        HumanMessage(
            content="I need current information about the latest developments in artificial intelligence in 2024.\nSpecifically, I'm interested in:\n1. Major AI model releases in 2024\n2. Recent breakthroughs in AI research\n3. Current AI safety developments\n\nPlease search for the most recent information and provide a comprehensive summary.",
        )
    ])
    
    logger.info("🚀 Starting direct tools agent execution...")
    
    # Execute the subgraph
    event_count = 0
    assistant_chunks = []
    tool_executions = []
    final_response = None
    
    try:
        async for event in subgraph.graph.astream(initial_state, {"recursion_limit": 15}):
            event_count += 1
            logger.info(f"📨 Event {event_count}: {type(event).__name__}")
            
            if isinstance(event, dict):
                for node_name, node_output in event.items():
                    logger.info(f"   Node: {node_name}")
                    
                    if 'messages' in node_output:
                        for msg in node_output['messages']:
                            if hasattr(msg, 'content'):
                                content = str(msg.content)[:200] + ("..." if len(str(msg.content)) > 200 else "")
                                logger.info(f"   Message content: {content}")
                                
                                # Look for tool calls vs responses
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    tool_executions.append({
                                        'tool_calls': msg.tool_calls,
                                        'content': msg.content
                                    })
                                    logger.info(f"   🛠️ Tool calls: {len(msg.tool_calls)}")
                                else:
                                    # This is a response, not tool calls
                                    assistant_chunks.append({
                                        'content': msg.content,
                                        'length': len(str(msg.content))
                                    })
                                    final_response = msg.content
                                    
            if event_count > 20:  # Safety limit
                logger.warning("⚠️ Event limit reached, stopping")
                break
                
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze results
    logger.info("📊 Analysis Results:")
    logger.info(f"   📝 Total events: {event_count}")
    logger.info(f"   🛠️ Tool executions: {len(tool_executions)}")
    logger.info(f"   📋 Assistant response chunks: {len(assistant_chunks)}")
    
    if final_response:
        logger.info(f"   ✅ Final response length: {len(str(final_response))} characters")
        logger.info(f"   📄 Response preview: {str(final_response)[:300]}...")
        
        # Check if response contains actual AI developments summary
        response_text = str(final_response).lower()
        has_summary_indicators = any(indicator in response_text for indicator in [
            "ai developments", "model releases", "breakthroughs", "safety", "2024",
            "artificial intelligence", "summary", "research", "google", "openai"
        ])
        
        if has_summary_indicators:
            logger.info("   ✅ Response appears to contain AI developments summary")
        else:
            logger.warning("   ⚠️ Response may not contain expected AI summary content")
            
    else:
        logger.error("   ❌ No final response generated")
    
    if tool_executions:
        logger.info("   🛠️ Tool execution details:")
        for i, exec_detail in enumerate(tool_executions):
            logger.info(f"      Tool set {i+1}: {len(exec_detail['tool_calls'])} calls")
    
    logger.info("✅ Test completed")

if __name__ == "__main__":
    asyncio.run(test_tools_agent_direct())