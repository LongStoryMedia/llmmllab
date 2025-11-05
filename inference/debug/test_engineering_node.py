"""
Simple test to verify engineering node gets user query in isolation
"""

import asyncio
from models import UserConfig, Message
from composer.nodes.agents.engineering import EngineeringAgentNode
from composer.graph.state import WorkflowState
from db import storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_engineering_node")

async def test_engineering_node():
    """Test engineering node directly"""
    
    logger.info("🔧 Testing Engineering Node Directly")
    
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
            
        # Create test query message
        test_query = "How can I implement JWT authentication in FastAPI with user registration and login endpoints?"
        
        from models import MessageRole, MessageContent, MessageContentType
        test_message = Message(
            role=MessageRole.USER,
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text=test_query,
                url=None
            )]
        )
        
        # Create test state with required fields for engineering node
        from models import IntentAnalysis, WorkflowType, ComplexityLevel, RequiredCapability, ComputationalRequirement
        
        # Create proper mock intent classification (engineering node requires this)
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
        
        # Create engineering agent - need pipeline factory, profile, metadata, tool storage
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
        
        logger.info(f"📝 Test query: {test_query[:50]}...")
        logger.info("🤖 Calling engineering node...")
        
        # Call the node
        result = await engineering_node(test_state)
        
        logger.info(f"🔍 Engineering node result type: {type(result)}")
        
        # Since it returns WorkflowState directly, check the messages
        if hasattr(result, 'messages'):
            original_count = len(test_state.messages)
            result_count = len(result.messages)
            logger.info(f"🔍 Original messages: {original_count}, Result messages: {result_count}")
            
            if result_count > original_count:
                new_messages = result.messages[original_count:]
                logger.info(f"🔍 Found {len(new_messages)} new messages from engineering node")
                
                for i, msg in enumerate(new_messages):
                    print(f"\n{'='*80}")
                    print(f"NEW MESSAGE {i+1} FROM ENGINEERING NODE:")
                    print(f"{'='*80}")
                    print(f"Type: {msg.type}")
                    print(f"Content: {msg.content}")
                    print(f"{'='*80}")
                    
                    # Check if response is generic
                    content = msg.content.lower()
                    if "hello" in content and "here to help" in content and len(content) < 200:
                        logger.warning("⚠️  Response appears to be generic greeting")
                    else:
                        logger.info("✅ Response appears to be technical content")
                return  # Exit early since we found the messages
            else:
                logger.error("❌ No new messages generated by engineering node")
                # Debug: Let's see what messages are actually there
                logger.info("🔍 Debugging message contents:")
                for i, msg in enumerate(result.messages):
                    logger.info(f"  Message {i}: type={msg.type}, content_length={len(msg.content)}")
                    logger.info(f"  Content preview: {msg.content[:100]}...")
                    
                    # Show the full engineering response if it's an AI message
                    if msg.type == "ai" and len(msg.content) > 100:
                        print(f"\n{'='*80}")
                        print(f"FULL ENGINEERING RESPONSE (Message {i}):")
                        print(f"{'='*80}")
                        print(msg.content)
                        print(f"{'='*80}")
                        
                        # Check if response is generic
                        content = msg.content.lower()
                        if "hello" in content and "here to assist" in content and len(content) < 1000:
                            logger.warning("⚠️  Response appears to be generic greeting despite 764 chars")
                        else:
                            logger.info("✅ Response appears to be technical content")
                        return
                
                # Check if there's a message that's been modified in place
                if len(result.messages) == len(test_state.messages):
                    for i, (orig, res) in enumerate(zip(test_state.messages, result.messages)):
                        if orig.content != res.content:
                            logger.info(f"🔍 Message {i} was modified!")
                            print(f"\n{'='*80}")
                            print(f"MODIFIED MESSAGE {i} FROM ENGINEERING NODE:")
                            print(f"{'='*80}")
                            print(f"Type: {res.type}")
                            print(f"Content: {res.content}")
                            print(f"{'='*80}")
                            return
        
        # This is the old logic in case it does return a Command
        if hasattr(result, 'update'):
            logger.info(f"🔍 Result has update: {bool(result.update)}")
            if result.update:
                logger.info(f"🔍 Update keys: {list(result.update.keys())}")
        else:
            logger.info("🔍 Result has no update attribute")
        
        if hasattr(result, 'update') and result.update:
            if 'messages' in result.update:
                new_messages = result.update['messages']
                if new_messages:
                    response = new_messages[-1].content
                    
                    print(f"\n{'='*80}")
                    print("ENGINEERING NODE RESPONSE:")
                    print(f"{'='*80}")
                    print(response)
                    print(f"{'='*80}")
                    
                    # Check if response is generic
                    content = response.lower()
                    if "hello" in content and "here to help" in content and len(content) < 200:
                        logger.warning("⚠️  Response appears to be generic greeting")
                    else:
                        logger.info("✅ Response appears to be technical content")
                else:
                    logger.error("No new messages in result")
            else:
                logger.error("No messages in result update")
        else:
            logger.error("No update in result")
            
    except Exception as e:
        logger.error(f"Engineering node test failed: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_engineering_node())