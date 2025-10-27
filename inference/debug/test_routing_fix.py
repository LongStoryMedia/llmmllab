"""
Test for composer workflow routing fix.
Verifies that the should_continue_agent_loop function prevents infinite tool execution loops.
"""

import asyncio
import os
import uuid
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="routing_fix_test")


async def test_routing_fix():
    """Test the routing fix for duplicate agent executions."""
    try:
        # Initialize database connection
        from db import storage
        from models.message import Message
        from models.message_role import MessageRole
        from models.message_content import MessageContent, MessageContentType
        from models.conversation import Conversation
        from datetime import datetime
        from datetime import datetime, timezone

        # Build connection string from environment variables
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "lsm")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "llmmll")
        db_sslmode = os.getenv("DB_SSLMODE", "disable")

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"

        await storage.initialize(connection_string)

        if not storage.initialized:
            raise RuntimeError("Storage failed to initialize properly")

        logger.info("Database initialized successfully")

        # Initialize composer
        import composer
        await composer.initialize_composer()
        logger.info("Composer initialized successfully")

        # Create test user and config
        test_user_id = f"routing_test_{uuid.uuid4().hex[:8]}"
        
        # Create user in database
        async with storage.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO users (id) VALUES ($1) ON CONFLICT (id) DO NOTHING",
                test_user_id,
            )
        
        # Create conversation
        conversation = Conversation(
            id=0,  # Will be set by database
            user_id=test_user_id,
            title="Routing Fix Test Conversation",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        conversation_id = await storage.conversation.create_conversation(conversation)
        
        # Create initial user message
        query_text = "What are 3 major AI developments in 2024?"
        content_list = [MessageContent(type=MessageContentType.TEXT, text=query_text)]
        
        user_message = Message(
            id=None,
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content=content_list,
            created_at=datetime.now(timezone.utc),
        )
        
        await storage.message.add_message(user_message)
        
        logger.info("Testing with user_id: %s, conversation_id: %d", test_user_id, conversation_id)
        logger.info("Creating workflow and initial state...")

        workflow = await composer.compose_workflow(test_user_id)
        initial_state = await composer.create_initial_state(
            user_id=test_user_id,
            conversation_id=conversation_id
        )

        logger.info("Executing workflow...")
        print("=" * 60)

        message_count = 0
        tool_execution_count = 0
        content_chunks = []

        async for chunk in composer.execute_workflow(workflow, initial_state):
            message_count += 1
            
            # Track content and tool executions
            if chunk.get('content'):
                content_chunks.append(chunk['content'])
                print(chunk['content'], end='')
            
            # Look for signs of tool execution in chunk data
            if 'tool_call' in str(chunk) or 'search' in str(chunk).lower():
                tool_execution_count += 1
            
            # Safety net to prevent infinite loops during testing
            if message_count > 200:
                print("\n[SAFETY STOP - message count exceeded 200]")
                break

        print("\n" + "=" * 60)
        logger.info("Test completed:")
        logger.info("  Total message chunks: %d", message_count)
        logger.info("  Tool execution indicators: %d", tool_execution_count)
        logger.info("  Content length: %d characters", len(''.join(content_chunks)))

        # Check if we have a reasonable response (not infinite loop)
        full_content = ''.join(content_chunks)
        if message_count < 200 and len(full_content) > 100:
            logger.info("✅ Routing fix appears to be working - finite response generated")
            return True
        else:
            logger.error("❌ Possible infinite loop or no content generated")
            return False

    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False
    finally:
        # Clean up test data and resources
        try:
            from db import storage
            if 'test_user_id' in locals() and storage and storage.pool:
                async with storage.pool.acquire() as conn:
                    # Clean up in reverse order of creation
                    await conn.execute("DELETE FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE user_id = $1)", test_user_id)
                    await conn.execute("DELETE FROM conversations WHERE user_id = $1", test_user_id)
                    await conn.execute("DELETE FROM users WHERE id = $1", test_user_id)
                logger.info("Test data cleaned up successfully")
        except Exception as cleanup_error:
            logger.warning("Cleanup failed: %s", str(cleanup_error))
        
        try:
            await composer.shutdown_composer()
        except Exception:
            pass


if __name__ == "__main__":
    result = asyncio.run(test_routing_fix())
    print(f"\nRouting fix test result: {'PASSED' if result else 'FAILED'}")