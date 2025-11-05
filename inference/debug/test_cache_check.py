#!/usr/bin/env python3
"""
Test with different query to check if model is caching responses
"""

import asyncio
from models import UserConfig, Message, MessageRole, MessageContent, MessageContentType
from composer.nodes.agents.engineering import EngineeringAgentNode
from composer.graph.state import WorkflowState
from db import storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_cache_check")

async def test_different_query():
    """Test with different query to check caching"""
    
    logger.info("🔧 Testing Engineering Node with Different Query")
    
    try:
        # Initialize database if needed
        if not storage.initialized:
            logger.info("📊 Initializing database connection...")
            from server.config import DB_CONNECTION_STRING
            if DB_CONNECTION_STRING:
                await storage.initialize(DB_CONNECTION_STRING)
                logger.info("✅ Database initialized")
            else:
                logger.error("No DB connection string available")
                return
        
        # Get user config
        user_config = await storage.user_config.get_user_config("test_user")
        if not user_config:
            logger.error("Failed to get user config")
            return
            
        # Use a DIFFERENT query to test caching
        test_query = "What are the best practices for database design in PostgreSQL?"
        
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text=test_query,
                url=None
            )]
        )
        
        # Create mock intent classification - engineering node requires this
        from models import IntentAnalysis, WorkflowType, ComplexityLevel, RequiredCapability, ComputationalRequirement
        
        intent_classification = [IntentAnalysis(
            workflow_type=WorkflowType.ENGINEERING,
            complexity_level=ComplexityLevel.MODERATE,
            required_capabilities=[RequiredCapability.TEXT_PROCESSING],
            domain_specificity=0.8,
            reusability_potential=0.6,
            confidence=0.95,
            tool_complexity_score=0.4,
            computational_requirements=ComputationalRequirement.MODERATE,
        )]
        
        test_state = WorkflowState(
            messages=[test_message],
            user_id="test_user",
            conversation_id=717,
            user_config=user_config,
            current_date="2025-11-04",
            intent_classification=intent_classification,
            current_user_message=test_message  # Engineering node also requires this
        )
        
        # Create engineering node - need to properly initialize the agent
        from runner import PipelineFactory
        from models.default_model_profiles import DEFAULT_ENGINEERING_PROFILE
        from db.dynamic_tool_storage import DynamicToolStorage
        
        # Create required components
        from runner import pipeline_factory
        engineering_profile = DEFAULT_ENGINEERING_PROFILE
        
        # Create node metadata
        from models import NodeMetadata
        engineering_node_metadata = NodeMetadata(
            node_id="test_engineering",
            node_name="test_engineering", 
            node_type="engineering"
        )
        
        # Create tool storage 
        tool_storage = DynamicToolStorage(storage.pool, None)  # get_query function not needed for test
        
        # Create engineering agent
        from composer.agents.engineering_agent import EngineeringAgent
        engineering_agent = EngineeringAgent(
            pipeline_factory,
            engineering_profile,
            engineering_node_metadata,
            tool_storage,
        )
        
        # Create engineering node
        engineering_node = EngineeringAgentNode(engineering_agent)
        
        logger.info(f"📝 Different test query: {test_query}")
        logger.info("🤖 Calling engineering node...")
        
        result = await engineering_node(test_state)
        
        logger.info(f"🔍 Engineering node result type: {type(result)}")
        logger.info(f"🔍 Original messages: {len(test_state.messages)}, Result messages: {len(result.messages)}")
        
        # Check if new messages were added
        if len(result.messages) > len(test_state.messages):
            new_message = result.messages[-1]
            logger.info(f"✅ New message generated: type={new_message.type}, length={len(new_message.content)}")
            logger.info("================================================================================")
            logger.info("FULL ENGINEERING RESPONSE (Different Query):")
            logger.info("================================================================================") 
            logger.info(new_message.content)
            logger.info("================================================================================")
            
            # Check if it's the same generic response
            if "Hello! I'm ready to help you with your technical questions" in new_message.content:
                logger.error("🚨 SAME GENERIC RESPONSE - This suggests model caching or systemic issue!")
            else:
                logger.info("✅ Different response generated - not a caching issue")
                
        else:
            logger.error("❌ No new messages generated by engineering node")
            logger.info("🔍 Debugging message contents:")
            for i, msg in enumerate(result.messages):
                logger.info(f"  Message {i}: type={msg.type}, content_length={len(msg.content)}")
                logger.info(f"  Content preview: {msg.content[:100]}...")
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_different_query())