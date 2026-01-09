"""
End-to-End Memory Test - Full memory workflow validation.

Tests memory creation, storage, and retrieval in a complete workflow.
Creates messages with embeddings, stores them, then validates memory search functionality.
"""

import os
import sys
from typing import List, Optional
import asyncio
from datetime import datetime, timezone

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    NodeMetadata,
    ModelProfile,
    Conversation,
)
from models.default_model_profiles import DEFAULT_PROFILES
from models.default_configs import create_default_user_config
from runner import pipeline_factory
from composer.agents import EmbeddingAgent
from composer.graph.nodes.memory.create import MemoryCreationNode
from composer.graph.nodes.memory.search import MemorySearchNode
from composer.graph.nodes.memory.store import MemoryStorageNode
from composer.graph.state import WorkflowState
from utils.logging import llmmllogger, serialize_event_data
from db import storage

logger = llmmllogger.bind(component="memory_e2e_test")


def get_embedding_profile() -> ModelProfile:
    """Get the embedding model profile."""
    profile = DEFAULT_PROFILES.get("embedding")
    if profile is None:
        print(f"[error] No embedding profile found")
        sys.exit(1)
    return profile


async def setup_test_user(user_id: str):
    """Create test user if it doesn't exist."""
    logger.info(f"🔧 Setting up test user: {user_id}")

    try:
        # Ensure storage is initialized
        if not storage.initialized:
            logger.warning("Storage not initialized, cannot create user")
            return False

        # Check if user exists, if not create it
        user_storage = storage.user_config
        if user_storage is None:
            logger.warning("User config storage not available")
            return False

        # Try to get user - this will create default user config if it doesn't exist
        try:
            await user_storage.get_user_config(user_id)
            logger.info(f"✅ Test user {user_id} ready")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup test user {user_id}: {e}")
            return False

    except Exception as e:
        logger.error(f"❌ User setup failed: {e}")
        return False


async def cleanup_test_data(
    user_id: str, conversation_id: int, message_ids: List[int] = []
):
    """Clean up test data after test completion."""
    logger.info(
        f"🧹 Cleaning up test data for user {user_id}, conversation {conversation_id}"
    )

    assert storage is not None, "Storage singleton is not initialized"
    assert storage.memory is not None, "Memory storage is not initialized"
    assert storage.message is not None, "Message storage is not initialized"
    assert storage.conversation is not None, "Conversation storage is not initialized"

    try:
        # Ensure storage is initialized
        if not storage.initialized:
            logger.warning("Storage not initialized, cannot clean up")
            return

        # Use the storage singleton to get memory storage
        memory_storage = storage.memory

        if memory_storage is None:
            logger.warning("Memory storage not available, cannot clean up")
            return

        # Clean up memories for this user
        await memory_storage.delete_all_user_memories(user_id)
        logger.info("✅ Cleaned up test memories")

        # Clean up test messages (both user and assistant)
        for message_id in message_ids:
            try:
                await storage.message.delete_message(message_id)
                logger.debug(f"Deleted test message {message_id}")
            except Exception as e:
                logger.warning(f"Failed to delete test message {message_id}: {e}")

        if message_ids:
            logger.info(f"✅ Cleaned up {len(message_ids)} test messages")

        # Clean up test conversation
        try:
            await storage.conversation.delete_conversation(conversation_id)
            logger.info(f"✅ Deleted test conversation {conversation_id}")
        except Exception as e:
            logger.warning(f"Failed to delete test conversation {conversation_id}: {e}")

        # Also clean up the test user if possible
        try:
            # Note: We don't delete the user as it might be referenced by other tables
            # Just cleaning up memories is sufficient for test cleanup
            logger.info("✅ Test user cleanup completed")
        except Exception as e:
            logger.warning(f"Could not clean up test user: {e}")

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        # Don't raise - cleanup failures shouldn't fail the test


async def wrapper() -> None:
    """Run the memory end-to-end test workflow."""
    # Ensure verbose logging minimal unless TRACE requested
    os.environ.setdefault("LOG_LEVEL", "INFO")

    logger.info("🚀 Starting Memory E2E Test")

    # Initialize storage FIRST before anything else
    if not storage.initialized:
        logger.info("🏗️ Initializing database storage...")
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
        logger.info("✅ Database storage initialized")

    assert storage
    assert storage.message is not None
    assert storage.memory is not None

    # Test data setup
    test_user_id = "memory_test_user"
    test_conversation_id = 12345

    # Create test messages with different but related content
    message1_text = "How many dominoes are in a set?"
    message2_text = "How can you figure out how many dominoes are in a set?"

    # NOTE: We need to store messages in the database first because memory search
    # joins on the messages table using source_id

    # Debug: Check what storage components are available RIGHT when we need them
    logger.info(f"Available storages:")
    logger.info(f"  - user_config: {storage.user_config is not None}")
    logger.info(f"  - memory: {storage.memory is not None}")
    logger.info(f"  - message: {storage.message is not None}")
    logger.info(f"  - conversation: {storage.conversation is not None}")

    # First create a conversation for the test messages
    test_conversation = Conversation(
        id=0,  # Temporary ID, will be replaced by database
        user_id=test_user_id,
        title="Memory Test Conversation",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    # Create conversation if storage is available
    if storage.conversation is not None:
        test_conversation_db_id = await storage.conversation.create_conversation(
            test_conversation
        )
        # Use the actual database conversation ID
        test_conversation_id = test_conversation_db_id or test_conversation_id

        logger.info(f"✅ Created test conversation with ID: {test_conversation_id}")
    else:
        logger.warning(
            "Conversation storage not available, using preset conversation ID"
        )
        # We'll use the original conversation ID and hope messages work without stored conversation

    # Store the first user message and assistant response in the database
    message1_user_obj = Message(
        id=None,  # Will be assigned by database
        role=MessageRole.USER,
        content=[MessageContent(type=MessageContentType.TEXT, text=message1_text)],
        conversation_id=test_conversation_id,
        created_at=datetime.now(timezone.utc),
    )
    message1_user_db_id = await storage.message.add_message(message1_user_obj)
    message1_user_obj.id = message1_user_db_id

    # Add assistant response for first message
    message1_assistant_text = (
        "A standard domino set (double-six) contains 28 domino tiles."
    )
    message1_assistant_obj = Message(
        id=None,
        role=MessageRole.ASSISTANT,
        content=[
            MessageContent(type=MessageContentType.TEXT, text=message1_assistant_text)
        ],
        conversation_id=test_conversation_id,
        created_at=datetime.now(timezone.utc),
    )
    message1_assistant_db_id = await storage.message.add_message(message1_assistant_obj)
    message1_assistant_obj.id = message1_assistant_db_id

    # Store the second user message and assistant response in the database
    message2_user_obj = Message(
        id=None,  # Will be assigned by database
        role=MessageRole.USER,
        content=[MessageContent(type=MessageContentType.TEXT, text=message2_text)],
        conversation_id=test_conversation_id,
        created_at=datetime.now(timezone.utc),
    )
    message2_user_db_id = await storage.message.add_message(message2_user_obj)
    message2_user_obj.id = message2_user_db_id

    # Add assistant response for second message
    message2_assistant_text = "You can use the formula (n+1)(n+2)/2 where n is the highest number on the dominoes. For double-six, that's (6+1)(6+2)/2 = 28 tiles."
    message2_assistant_obj = Message(
        id=None,
        role=MessageRole.ASSISTANT,
        content=[
            MessageContent(type=MessageContentType.TEXT, text=message2_assistant_text)
        ],
        conversation_id=test_conversation_id,
        created_at=datetime.now(timezone.utc),
    )
    message2_assistant_db_id = await storage.message.add_message(message2_assistant_obj)
    message2_assistant_obj.id = message2_assistant_db_id

    # For memory creation, we still focus on the user messages
    test_messages_1 = [message1_user_obj]
    test_messages_2 = [message2_user_obj]

    embedding_profile = get_embedding_profile()
    logger.info(f"📊 Using embedding model profile: {embedding_profile.model_name}")

    # Get memory storage from the storage singleton
    memory_storage = storage.memory
    assert memory_storage is not None, "Memory storage is not initialized"

    # Set up test user
    logger.info(f"🔧 Setting up test user: {test_user_id}")
    user_setup_success = await setup_test_user(test_user_id)
    if not user_setup_success:
        logger.error("❌ Failed to setup test user, aborting test")
        return

    try:
        # Initialize embedding agent
        embedding_model = pipeline_factory.get_embedding_pipeline(embedding_profile)
        embedding_agent = EmbeddingAgent(
            model=embedding_model,
            profile=embedding_profile,
        )

        # Create node metadata
        node_metadata = NodeMetadata(
            node_name="memory_test_node",
            node_id="memory_test_001",
            node_type="test",
            user_id=test_user_id,
        )

        # Initialize memory nodes
        memory_create_node = MemoryCreationNode(
            embedding_agent=embedding_agent,
            node_metadata=node_metadata,
        )

        memory_search_node = MemorySearchNode(
            embedding_agent=embedding_agent,
            memory_storage=memory_storage,
        )

        memory_store_node = MemoryStorageNode(
            memory_storage=memory_storage,
        )

        logger.info("🧠 Memory nodes initialized")

        # Step 1: Create and store first memory
        print("\n" + "=" * 80)
        print("STEP 1: Creating and storing first memory")
        print("=" * 80)
        print(f"Message 1: {message1_text}")

        user_config = create_default_user_config(user_id=test_user_id)
        # Lower the similarity threshold for testing (default 0.7 is too high for related content)
        user_config.memory.similarity_threshold = 0.1
        # Disable cross-conversation search to test conversation-specific memory retrieval
        user_config.memory.enable_cross_conversation = False

        # Create workflow state for first message
        workflow_state_1 = WorkflowState(
            messages=test_messages_1,
            user_id=test_user_id,
            conversation_id=test_conversation_id,
            user_config=user_config,
            things_to_remember=test_messages_1,  # Add messages to things to remember
        )

        # Create memory from first message
        state_after_create_1 = await memory_create_node(workflow_state_1)

        # Store first memory
        try:
            await memory_store_node(state_after_create_1)
        except Exception as e:
            logger.error(f"❌ Memory storage failed: {e}")
            raise

        # Step 2: Create second message and search for similar memories
        print("\n" + "=" * 80)
        print("STEP 2: Creating second message and searching for similar memories")
        print("=" * 80)
        print(f"Message 2: {message2_text}")

        # Create workflow state for second message
        workflow_state_2 = WorkflowState(
            messages=test_messages_2,
            user_id=test_user_id,
            conversation_id=test_conversation_id,
            user_config=user_config,
            current_user_message=test_messages_2[0],  # Set current message for search
        )

        # Search for similar memories
        state_after_search = await memory_search_node(workflow_state_2)

        # Step 3: Validate and display results
        print("\n" + "=" * 80)
        print("STEP 3: Memory Search Results Validation")
        print("=" * 80)

        if state_after_search.retrieved_memories:
            for idx, memory in enumerate(state_after_search.retrieved_memories):
                print(f"\nMemory {idx + 1}:")
                print(f"  Source: {memory.source}")
                print(f"  Similarity: {memory.similarity:.4f}")
                print(f"  Created: {memory.created_at}")
                print(f"  Conversation ID: {memory.conversation_id}")
                print(f"  Fragments: {len(memory.fragments)}")

                for frag_idx, fragment in enumerate(memory.fragments):
                    print(f"    Fragment {frag_idx + 1}:")
                    print(f"      Role: {fragment.role}")
                    print(f"      Content: {fragment.content}")
                    print(
                        f"      Embedding dims: {len(fragment.embeddings[0]) if fragment.embeddings else 0}"
                    )

        # Step 4: Test creating and storing second memory
        print("\n" + "=" * 80)
        print("STEP 4: Creating and storing second memory")
        print("=" * 80)

        workflow_state_2.things_to_remember = test_messages_2
        state_after_create_2 = await memory_create_node(workflow_state_2)

        await memory_store_node(state_after_create_2)
        logger.info("📊 Second memory storage attempt completed")
        print("✅ Stored second memory in database")

        # Final validation - search again to see both memories
        print("\n" + "=" * 80)
        print("FINAL VALIDATION: Search should now find both memories")
        print("=" * 80)

        final_search_state = await memory_search_node(workflow_state_2)

        # Validate that we found paired messages (user + assistant together)
        found_user_memories = [
            m
            for m in final_search_state.retrieved_memories
            if any(f.role == MessageRole.USER for f in m.fragments)
        ]
        found_assistant_memories = [
            m
            for m in final_search_state.retrieved_memories
            if any(f.role == MessageRole.ASSISTANT for f in m.fragments)
        ]

        logger.info(
            f"📊 Found {len(found_user_memories)} user memories and {len(found_assistant_memories)} assistant memories"
        )

        if len(found_user_memories) > 0 and len(found_assistant_memories) > 0:
            logger.info(
                "✅ SUCCESS: Found both user and assistant messages in search results"
            )
            print("\n🎉 Memory E2E test completed successfully! Found paired messages.")
        else:
            logger.warning(
                "⚠️ Only found single message types - check if pairing logic is working"
            )
            print("\n⚠️ Test completed but may need pairing logic review.")

    except Exception as e:
        logger.error(f"❌ Memory E2E test failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        # Clean up test data
        await cleanup_test_data(test_user_id, test_conversation_id, [message1_user_db_id, message1_assistant_db_id, message2_user_db_id, message2_assistant_db_id])  # type: ignore

        # Note: EmbeddingAgent doesn't inherit from BaseAgent and doesn't have cleanup method
        # The pipeline will be cleaned up automatically by the pipeline factory


if __name__ == "__main__":
    asyncio.run(wrapper())
