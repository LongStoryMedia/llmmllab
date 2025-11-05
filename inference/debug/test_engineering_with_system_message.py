#!/usr/bin/env python3
"""
Test engineering agent with direct system message to bypass system prompt issues
"""

import asyncio
from models import UserConfig, Message, MessageRole, MessageContent, MessageContentType
from composer.nodes.agents.engineering import EngineeringAgentNode
from composer.graph.state import WorkflowState
from db import storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_engineering_with_system_message")

async def test_engineering_with_system_message():
    """Test engineering node with explicit system message"""
    
    logger.info("🔧 Testing Engineering Node with System Message")
    
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
            
        # Create test query message with explicit system message
        test_query = "How can I implement JWT authentication in FastAPI with user registration and login endpoints?"
        
        # Add a system message first
        system_message = Message(
            role=MessageRole.SYSTEM,
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text="You are an expert engineering assistant. Answer the user's technical question directly with comprehensive code examples and practical implementation details. Do not ask for clarification - provide a complete solution.",
                url=None
            )]
        )
        
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
            messages=[system_message, test_message],  # Include system message
            user_id="test_user",
            conversation_id=717,
            user_config=user_config,
            current_date="2025-11-04",
            intent_classification=intent_classification,
            current_user_message=test_message  # Engineering node also requires this
        )
        
        logger.info(f"📝 Test query: {test_query}")
        logger.info("🤖 Calling engineering node with explicit system message...")
        
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
        
        result = await engineering_node(test_state)
        
        logger.info(f"🔍 Engineering node result type: {type(result)}")
        logger.info(f"🔍 Original messages: {len(test_state.messages)}, Result messages: {len(result.messages)}")
        
        # Check if new messages were added
        if len(result.messages) > len(test_state.messages):
            new_message = result.messages[-1]
            logger.info(f"✅ New message generated: type={new_message.type}, length={len(new_message.content)}")
            logger.info("================================================================================")
            logger.info("FULL ENGINEERING RESPONSE (with explicit system message):")
            logger.info("================================================================================") 
            logger.info(new_message.content)
            logger.info("================================================================================")
            
            # Check if response looks technical
            response_lower = new_message.content.lower()
            technical_indicators = ['jwt', 'fastapi', 'authentication', 'token', 'endpoint', 'code', 'import', 'def', 'class', 'install', 'pip', 'from', 'router']
            found_indicators = [term for term in technical_indicators if term in response_lower]
            
            if found_indicators:
                logger.info(f"✅ Response contains technical content: {found_indicators}")
            else:
                logger.info("❌ Response does not appear to contain technical content")
                
            # Check for generic indicators
            generic_indicators = ["i'm ready to help", "what would you like", "please share", "how can i help", "what specific"]
            found_generic = [term for term in generic_indicators if term in response_lower]
            
            if found_generic:
                logger.info(f"⚠️  Response still contains generic language: {found_generic}")
            else:
                logger.info("✅ Response does not contain obvious generic language")
                
        else:
            logger.error("❌ No new messages generated by engineering node")
            logger.info("🔍 Debugging message contents:")
            for i, msg in enumerate(result.messages):
                logger.info(f"  Message {i}: type={msg.type}, content_length={len(msg.content)}")
                logger.info(f"  Content preview: {msg.content[:100]}...")
                
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(test_engineering_with_system_message())