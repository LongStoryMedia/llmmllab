#!/usr/bin/env python3
"""
Simple diagnostic test to check why ChatOpenAI pipeline detection is failing.
"""

import sys
import os
import asyncio
sys.path.append('/app')

from composer.agents.chat_agent import ChatAgent
from composer.tools.registry import ToolRegistry
from models import ModelProfile, ModelParameters, NodeMetadata
from models.default_configs import DEFAULT_USER_CONFIG
from runner.pipeline_factory import PipelineFactory
from utils.logging import llmmllogger

async def test_chatopenai_detection():
    logger = llmmllogger.bind(component="ChatOpenAIDetectionTest")
    
    try:
        logger.info("🚀 Starting ChatOpenAI detection test")
        
        # Create test profile for qwen3-vl-30b-a3b-thinking
        test_profile = ModelProfile(
            id="00000000-0000-0000-0000-000000000001",
            user_id="test_user",
            name="Primary (Default)",
            description="Test profile",
            model_name="qwen3-vl-30b-a3b-thinking",
            parameters=ModelParameters(
                num_ctx=65536,
                repeat_penalty=1.1,
                temperature=0.6,
                seed=-1,
                stop=["<|im_end|>"],
                num_predict=-1,
                top_k=20,
                top_p=0.95,
                min_p=0.01,
                think=True,
                max_tokens=400000,
                batch_size=128,
            ),
            system_prompt="Test prompt",
        )
        
        # Create node metadata
        node_metadata = NodeMetadata(
            node_id="test_001",
            node_name="test_node", 
            node_type="test",
            user_id="test_user",
            conversation_id=123
        )
        
        # Create pipeline factory
        pipeline_factory = PipelineFactory({})
        
        # Create chat agent
        chat_agent = ChatAgent(
            pipeline_factory=pipeline_factory,
            profile=test_profile,
            node_metadata=node_metadata
        )
        
        logger.info("✅ ChatAgent created successfully")
        
        # Check if agent has _pipeline attribute
        has_pipeline_attr = hasattr(chat_agent, '_pipeline')
        logger.info(f"📋 Agent has _pipeline attribute: {has_pipeline_attr}")
        
        if has_pipeline_attr:
            logger.info(f"📋 Agent _pipeline value: {chat_agent._pipeline}")
            logger.info(f"📋 Agent _pipeline type: {type(chat_agent._pipeline)}")
            
            if chat_agent._pipeline:
                has_get_chat_model = hasattr(chat_agent._pipeline, 'get_chat_model')
                logger.info(f"📋 Pipeline has get_chat_model method: {has_get_chat_model}")
                
                if has_get_chat_model:
                    logger.info("✅ Pipeline has get_chat_model method!")
                    try:
                        chat_model = chat_agent._pipeline.get_chat_model()
                        logger.info(f"📋 get_chat_model() returned: {type(chat_model)}")
                        logger.info(f"📋 ChatModel: {chat_model}")
                    except Exception as e:
                        logger.error(f"❌ Error calling get_chat_model(): {e}")
                else:
                    logger.warning("❌ Pipeline does not have get_chat_model method")
                    
                    # Check what methods it does have
                    methods = [method for method in dir(chat_agent._pipeline) if not method.startswith('_')]
                    logger.info(f"📋 Available pipeline methods: {methods[:10]}...")  # Show first 10
            else:
                logger.warning("❌ Agent _pipeline is None")
        else:
            logger.warning("❌ Agent does not have _pipeline attribute")
            
        # Also test pipeline creation directly
        logger.info("🔧 Testing direct pipeline creation...")
        try:
            from runner.pipelines.llamacpp.langchain_chatopenai_pipeline import LangChainChatOpenAIPipeline
            from utils.model_loader import ModelLoader
            
            # Get model
            model_loader = ModelLoader()
            available_models = model_loader.get_available_models()
            model = available_models.get("qwen3-vl-30b-a3b-thinking")
            
            if model:
                logger.info(f"📋 Found model: {model.name}")
                
                # Try creating LangChain pipeline directly
                pipeline = LangChainChatOpenAIPipeline(model, test_profile)
                logger.info(f"✅ Created LangChain pipeline: {type(pipeline)}")
                
                has_get_chat_model = hasattr(pipeline, 'get_chat_model')
                logger.info(f"📋 LangChain pipeline has get_chat_model: {has_get_chat_model}")
                
                if has_get_chat_model:
                    try:
                        chat_model = pipeline.get_chat_model()
                        logger.info(f"✅ get_chat_model() works: {type(chat_model)}")
                    except Exception as e:
                        logger.error(f"❌ get_chat_model() failed: {e}")
                
            else:
                logger.error("❌ Model qwen3-vl-30b-a3b-thinking not found")
                logger.info(f"📋 Available models: {list(available_models.keys())}")
                
        except Exception as e:
            logger.error(f"❌ Error testing direct pipeline creation: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info("🎉 ChatOpenAI detection test completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_chatopenai_detection())