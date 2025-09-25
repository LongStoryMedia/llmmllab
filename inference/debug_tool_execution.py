#!/usr/bin/env python3
"""
Debug Tool Execution Script

This script tests the actual tool execution to see what's happening
with the web_search tool and response generation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

# Configure logging for debug
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_tool_execution():
    """Test the actual tool execution to see what happens."""
    try:
        # Initialize database and pipeline factory
        from server.db import storage
        from runner.pipeline_factory import pipeline_factory
        from server.tools.integration import get_tools
        from server.services.context import ConversationContext
        from models.user_config import UserConfig
        from models.default_configs import create_default_user_config
        from runner.pipelines.run import stream_pipeline
        
        logger.info('🔧 Initializing infrastructure for focused test...')
        
        # Database connection
        import os
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_user = os.getenv('DB_USER', 'postgres')
        db_password = os.getenv('DB_PASSWORD', '')
        db_name = os.getenv('DB_NAME', 'llmmllab')
        db_sslmode = os.getenv('DB_SSLMODE', 'disable')
        connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}'
        await storage.initialize(connection_string)
        logger.info('✅ Database connected')
        
        # Create test entities
        test_user_id = f'test_tool_debug_{uuid.uuid4().hex[:8]}'
        test_model_profile_id = uuid.uuid4()
        
        # Create model profile
        from models.model_profile import ModelProfile
        from models.model_parameters import ModelParameters
        
        model_profile = ModelProfile(
            id=test_model_profile_id,
            user_id=test_user_id,
            name='Tool Debug Profile',
            description='Debug profile for tool execution',
            model_name='openai-gpt-oss-20b-uncensored-q5_1',
            type=0,  # Chat type
            parameters=ModelParameters(temperature=0.7, top_p=0.9, max_tokens=1000, flash_attention=True),
            system_prompt='You are a helpful AI assistant with access to tools. When users ask for current information, use the web_search tool.'
        )
        
        created_profile = await storage.model_profile.create_model_profile(model_profile)
        logger.info(f'✅ Model profile created: {created_profile.id}')
        
        # Create conversation
        # Ensure user exists first
        async with storage.pool.acquire() as conn:
            await conn.execute(
                storage.get_query("user.ensure_user"), test_user_id
            )
        logger.info(f'✅ Ensured user exists: {test_user_id}')
        
        # Create real conversation
        test_conversation_id = await storage.conversation.create_conversation(
            user_id=test_user_id,
            title='Tool Debug Conversation',
        )
        
        if not test_conversation_id:
            raise Exception('Failed to create conversation')
        
        logger.info(f'✅ Conversation created: {test_conversation_id}')
        
        # Create user message
        from models.message import Message
        from models.message_role import MessageRole
        from models.message_content import MessageContent, MessageContentType
        
        user_msg = Message(
            id=None,  # Will be set by database
            conversation_id=test_conversation_id,
            role=MessageRole.USER,
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text='What are the latest quantum computing breakthroughs in 2024?'
            )],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        
        # Add message to database
        message_id = await storage.message.add_message(user_msg)
        if not message_id:
            raise Exception('Failed to create user message')
        
        logger.info(f'✅ User message created: {message_id}')
        
        # Get tools
        user_config = create_default_user_config(test_user_id)
        await storage.user_config.update_user_config(test_user_id, user_config)
        
        conversation_ctx = ConversationContext(
            conversation_id=test_conversation_id, 
            user_config=user_config
        )
        conversation_ctx.current_user_message = user_msg
        
        tools = []
        async for tool_result in get_tools(conversation_ctx):
            if isinstance(tool_result, list):
                tools.extend(tool_result)
                break
        
        logger.info(f'🛠️ Available tools: {[tool.name for tool in tools]}')
        
        # Get pipeline
        from models.chat_response import ChatResponse
        pipeline = pipeline_factory.get_pipeline(
            profile=created_profile, expected_type=ChatResponse
        )
        
        if not pipeline:
            raise Exception('Failed to get pipeline')
        logger.info('✅ Pipeline obtained')
        
        # Execute pipeline with streaming
        messages = [user_msg]
        response_chunks = []
        tool_calls_found = []
        
        logger.info('🚀 Starting pipeline execution...')
        chunk_count = 0
        
        async for chunk in stream_pipeline(messages, pipeline, tools):
            chunk_count += 1
            response_chunks.append(chunk)
            
            # Log every chunk to see exactly what's happening
            logger.info(f'📦 Chunk {chunk_count}: Type={type(chunk)}')
            
            if hasattr(chunk, 'content') and chunk.content:
                logger.info(f'📄 Content: {chunk.content[:200]}...')
            else:
                logger.info(f'📄 No content attribute or empty content')
            
            # Check for tool calls
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls_found.extend(chunk.tool_calls)
                logger.info(f'🛠️ Tool call detected: {chunk.tool_calls}')
            
            # Check content for tool execution patterns
            if hasattr(chunk, 'content') and chunk.content:
                content = str(chunk.content).lower()
                if 'web_search' in content:
                    logger.info(f'🔍 Web search detected in chunk {chunk_count}')
                if 'quantum' in content:
                    logger.info(f'🔬 Search results detected in chunk {chunk_count}')
                if 'breakthrough' in content:
                    logger.info(f'🎯 Breakthrough content detected in chunk {chunk_count}')
        
        logger.info(f'✅ Pipeline completed: {len(response_chunks)} chunks, {len(tool_calls_found)} tool calls')
        
        # Analyze final result
        full_content = ""
        for chunk in response_chunks:
            if hasattr(chunk, 'content') and chunk.content:
                full_content += str(chunk.content)
        
        logger.info(f'📊 Final analysis:')
        logger.info(f'   - Total content length: {len(full_content)}')
        logger.info(f'   - Contains "web_search": {"web_search" in full_content.lower()}')
        logger.info(f'   - Contains "quantum": {"quantum" in full_content.lower()}')
        logger.info(f'   - Contains "breakthrough": {"breakthrough" in full_content.lower()}')
        logger.info(f'   - Tool calls found: {len(tool_calls_found)}')
        
        if full_content:
            logger.info(f'📄 Full content preview: {full_content[:500]}...')
        
        # Cleanup
        try:
            await storage.message.delete_message(message_id, test_conversation_id)
            await storage.conversation.delete_conversation(test_conversation_id)
            await storage.model_profile.delete_model_profile(test_model_profile_id, test_user_id)
            logger.info('🧹 Cleanup completed')
        except Exception as cleanup_error:
            logger.warning(f'⚠️ Cleanup error: {cleanup_error}')
        
    except Exception as e:
        logger.error(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_tool_execution())