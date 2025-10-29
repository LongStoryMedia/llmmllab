#!/usr/bin/env python3
"""
Individual Node/Agent/Subgraph Testing Framework

Provides utilities to test LangGraph nodes, agents, and subgraphs in isolation
without running the full workflow. Useful for debugging and development.

Usage:
    # Test memory node in isolation
    python -m debug.test_individual_components memory

    # Test tools agent subgraph
    python -m debug.test_individual_components tools_agent

    # Test chat node with custom input
    python -m debug.test_individual_components chat_node --input "Tell me about AI"
"""

import asyncio
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path

# Import composer components for testing
from composer.graph.state import WorkflowState
from composer.nodes.memory.memory_creation import MemoryCreationNode
from composer.nodes.agents.chat_node import ChatNode  
from composer.graph.subgraphs.tools_agent import ToolsAgentSubgraph
from composer.agents.embedding_agent import EmbeddingAgent
from composer.agents.chat_agent import ChatAgent
from composer.core.service import ComposerService
from db import storage
from models import (
    Message, MessageContent, MessageContentType, MessageRole,
    UserConfig, LangChainMessage
)
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="individual_component_tester")


class IndividualComponentTester:
    """Framework for testing individual LangGraph components."""
    
    def __init__(self):
        self.composer_service = None
        
    async def initialize(self):
        """Initialize testing environment."""
        logger.info("Initializing individual component testing framework...")
        
        # Initialize composer service
        self.composer_service = ComposerService()
        
        # Initialize database connection pool
        await storage.initialize()
        
        logger.info("✅ Testing framework initialized")
        
    async def cleanup(self):
        """Clean up resources."""
        if self.composer_service:
            await self.composer_service.shutdown()
        logger.info("✅ Testing framework cleaned up")
        
    async def create_test_state(
        self, 
        user_input: str = "Test input",
        user_id: str = "test_user",
        conversation_id: int = 999999
    ) -> WorkflowState:
        """Create a test WorkflowState for component testing."""
        
        # Create test user config
        user_config = await storage.get_service(storage.user_config).get_user_config(user_id)
        
        # Create test message
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(type=MessageContentType.TEXT, text=user_input)],
            conversation_id=conversation_id
        )
        
        # Convert to LangChain format
        langchain_messages = [
            LangChainMessage(content=user_input, type="user")
        ]
        
        # Create test state
        state = WorkflowState(
            messages=langchain_messages,
            user_id=user_id,
            user_config=user_config,
            conversation_id=conversation_id,
            current_user_message=LangChainMessage(content=user_input, type="user"),
            checkpoint_metadata={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "test": True
            }
        )
        
        return state
        
    async def test_memory_node(self, user_input: str = "Remember this: AI is fascinating") -> Dict[str, Any]:
        """Test MemoryCreationNode in isolation."""
        logger.info("🧠 Testing MemoryCreationNode...")
        
        try:
            # Initialize embedding agent
            embedding_agent = EmbeddingAgent()
            
            # Create memory node
            memory_node = MemoryCreationNode(embedding_agent)
            
            # Create test state with things to remember
            state = await self.create_test_state(user_input)
            state.things_to_remember = [
                Message(
                    role=MessageRole.USER,
                    content=[MessageContent(type=MessageContentType.TEXT, text=user_input)],
                    conversation_id=state.conversation_id
                )
            ]
            
            # Execute node
            result_state = await memory_node(state)
            
            logger.info(f"✅ Memory node test completed. Created memories: {len(result_state.created_memories)}")
            
            return {
                "success": True,
                "input": user_input,
                "created_memories_count": len(result_state.created_memories),
                "created_memories": [
                    {
                        "fragments": len(memory.fragments) if memory.fragments else 0,
                        "source": memory.source.value if memory.source else None
                    }
                    for memory in result_state.created_memories
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Memory node test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_chat_node(self, user_input: str = "What is artificial intelligence?") -> Dict[str, Any]:
        """Test ChatNode in isolation."""
        logger.info("💬 Testing ChatNode...")
        
        try:
            # Create chat agent
            chat_agent = ChatAgent()
            
            # Create chat node  
            chat_node = ChatNode(chat_agent)
            
            # Create test state
            state = await self.create_test_state(user_input)
            
            # Execute node
            result_state = await chat_node(state)
            
            logger.info("✅ Chat node test completed")
            
            return {
                "success": True,
                "input": user_input,
                "output_messages_count": len(result_state.messages),
                "last_message": result_state.messages[-1].content if result_state.messages else None
            }
            
        except Exception as e:
            logger.error(f"❌ Chat node test failed: {e}")
            return {"success": False, "error": str(e)}
            
    async def test_tools_agent(self, user_input: str = "What time is it?") -> Dict[str, Any]:
        """Test ToolsAgentSubgraph in isolation."""
        logger.info("🔧 Testing ToolsAgentSubgraph...")
        
        try:
            # Create tools agent subgraph
            tools_agent = ToolsAgentSubgraph()
            
            # Create test state
            state = await self.create_test_state(user_input)
            
            # Transform to tools state
            tools_state = tools_agent.transform_to_tools_state(state)
            
            logger.info("✅ Tools agent transformation test completed")
            
            return {
                "success": True,
                "input": user_input,
                "tools_state_messages": len(tools_state["messages"]),
                "user_id": tools_state["user_id"],
                "conversation_id": tools_state["conversation_id"]
            }
            
        except Exception as e:
            logger.error(f"❌ Tools agent test failed: {e}")
            return {"success": False, "error": str(e)}


async def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test individual LangGraph components")
    parser.add_argument("component", choices=["memory", "chat_node", "tools_agent"], help="Component to test")
    parser.add_argument("--input", default=None, help="Custom input text for testing")
    parser.add_argument("--output", default=None, help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = IndividualComponentTester()
    await tester.initialize()
    
    try:
        # Run appropriate test
        result = {}
        
        if args.component == "memory":
            test_input = args.input or "Remember this: LangGraph is a great framework for building AI agents"
            result = await tester.test_memory_node(test_input)
            
        elif args.component == "chat_node":
            test_input = args.input or "Explain the benefits of modular AI agent architectures"
            result = await tester.test_chat_node(test_input)
            
        elif args.component == "tools_agent":
            test_input = args.input or "What is the current time and date?"
            result = await tester.test_tools_agent(test_input)
            
        # Output results
        print("\n" + "="*80)
        print(f"🧪 INDIVIDUAL COMPONENT TEST RESULTS: {args.component.upper()}")
        print("="*80)
        print(json.dumps(result, indent=2))
        print("="*80)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(json.dumps(result, indent=2))
            logger.info(f"📄 Results saved to {output_path}")
            
    finally:
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())