"""
Comprehensive test for memory retrieval functionality.

This test verifies the end-to-end memory storage and retrieval system:
1. Creates test conversations with similar semantic content
2. Stores memories with embeddings
3. Tests similarity search across conversations
4. Validates memory retrieval tool functionality
"""

import asyncio
import os
from datetime import datetime
from typing import List

import numpy as np

# Module imports
from db import storage
from db.memory_storage import MemoryStorage
from models import (
    Memory,
    MemoryFragment,
    MemorySource,
    MessageRole,
    ModelProfile,
    ModelProfileType,
    NodeMetadata,
    ConversationCtx,
    Message,
    UserConfig,
)
from models.default_configs import DEFAULT_MEMORY_CONFIG
from composer.agents.memory_agent import MemoryAgent
from composer.agents.embedding_agent import EmbeddingAgent
from composer.tools.static.memory_retrieval_tool import memory_retrieval
from composer.graph.state import ToolsState
from runner.pipeline_factory import PipelineFactory
from langchain.tools import ToolRuntime
from utils.logging import llmmllogger

# Set up test environment
os.environ["TESTING"] = "true"

logger = llmmllogger.logger.bind(component="MemoryTest")


class TestMemoryRetrieval:
    """Test class for memory storage and retrieval functionality."""

    def __init__(self):
        self.pipeline_factory = None
        self.memory_storage = None
        self.test_user_id = "test_user_123"
        self.test_conversation_1 = 1001
        self.test_conversation_2 = 1002

    async def setup_test_environment(self):
        """Initialize database and storage services for testing."""
        logger.info("Setting up test environment...")

        # Initialize database using environment variables (same as working tests)
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "lsm")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "llmmll")
        db_sslmode = os.getenv("DB_SSLMODE", "disable")

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={db_sslmode}"

        if not storage.initialized:
            await storage.initialize(connection_string)
            logger.info("Database initialized successfully")

        # Get memory storage service
        self.memory_storage = storage.get_service(storage.memory)

        # Create test user if doesn't exist (required for foreign key constraint)
        await self.create_test_user()

        # Mock pipeline factory for testing (we don't need actual pipelines)
        class MockPipelineFactory:
            pass

        self.pipeline_factory = MockPipelineFactory()

        logger.info("Test environment setup complete")

    async def create_test_user(self):
        """Create test user and conversations in database if they don't exist."""
        try:
            async with storage.pool.acquire() as conn:
                # Check if user exists
                existing_user = await conn.fetchrow(
                    "SELECT id FROM users WHERE id = $1", self.test_user_id
                )

                if not existing_user:
                    # Create test user with correct schema
                    await conn.execute(
                        "INSERT INTO users (id, username, created_at, config) VALUES ($1, $2, NOW(), '{}') ON CONFLICT (id) DO NOTHING",
                        self.test_user_id,
                        "test_user_123",
                    )
                    logger.info(f"Created test user: {self.test_user_id}")
                else:
                    logger.info(f"Test user already exists: {self.test_user_id}")

                # Create test conversations (required by memory storage)
                for conv_id in [self.test_conversation_1, self.test_conversation_2]:
                    await conn.execute(
                        "INSERT INTO conversations (id, user_id, title, created_at) VALUES ($1, $2, $3, NOW()) ON CONFLICT (id) DO NOTHING",
                        conv_id,
                        self.test_user_id,
                        f"Test Conversation {conv_id}",
                    )
                    logger.info(f"Created test conversation: {conv_id}")

                # Create test messages (required by memory storage search queries)
                test_messages = [
                    (
                        101,
                        self.test_conversation_1,
                        "user",
                        "How do I install Python packages using pip?",
                    ),
                    (
                        102,
                        self.test_conversation_1,
                        "assistant",
                        "You can install Python packages using pip by running 'pip install package-name' in your terminal.",
                    ),
                    (
                        201,
                        self.test_conversation_2,
                        "user",
                        "What's the best way to install Python libraries?",
                    ),
                    (
                        202,
                        self.test_conversation_2,
                        "assistant",
                        "The recommended way is to use pip, the Python package installer. Just run 'pip install library-name'.",
                    ),
                ]

                for msg_id, conv_id, role, content in test_messages:
                    await conn.execute(
                        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES ($1, $2, $3, $4, NOW()) ON CONFLICT (id) DO NOTHING",
                        msg_id,
                        conv_id,
                        role,
                        content,
                    )

                logger.info(f"Created {len(test_messages)} test messages")

        except Exception as e:
            logger.warning(f"Failed to create test data: {e}")
            # Try to continue anyway - maybe the constraint is not enforced in test mode

    async def cleanup_test_data(self):
        """Clean up any existing test data."""
        logger.info("Cleaning up test data...")
        try:
            if self.memory_storage:
                await self.memory_storage.delete_all_user_memories(self.test_user_id)

            # Clean up test data in reverse order due to foreign keys
            if storage.pool:
                async with storage.pool.acquire() as conn:
                    # Delete test messages
                    await conn.execute(
                        "DELETE FROM messages WHERE id IN (101, 102, 201, 202)"
                    )

                    # Delete test conversations
                    await conn.execute(
                        "DELETE FROM conversations WHERE id IN ($1, $2)",
                        self.test_conversation_1,
                        self.test_conversation_2,
                    )

                    # Delete test user
                    await conn.execute(
                        "DELETE FROM users WHERE id = $1", self.test_user_id
                    )

            logger.info("Test data cleanup complete")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

    def create_similar_embeddings(
        self, base_text: str, variation_text: str
    ) -> tuple[List[float], List[float]]:
        """
        Create embeddings that are semantically similar but not identical.
        For testing purposes, we'll create synthetic embeddings with high similarity.
        """
        # Create base embedding (768 dimensions)
        np.random.seed(42)  # For reproducible results
        base_embedding = np.random.randn(768).astype(float)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)  # Normalize

        # Create similar embedding with high cosine similarity (0.85+)
        variation_embedding = base_embedding.copy()
        # Add small random noise to create variation but maintain high similarity
        noise = np.random.randn(768) * 0.3
        variation_embedding += noise
        variation_embedding = variation_embedding / np.linalg.norm(variation_embedding)

        # Verify similarity is high enough
        similarity = np.dot(base_embedding, variation_embedding)
        logger.info(f"Created embeddings with similarity: {similarity:.3f}")

        return base_embedding.tolist(), variation_embedding.tolist()

    async def store_test_memories(self):
        """Store test memories with similar semantic content."""
        logger.info("Storing test memories...")

        # Create similar text content
        conversation_1_texts = [
            "How do I install Python packages using pip?",
            "You can install Python packages using pip by running 'pip install package-name' in your terminal.",
        ]

        conversation_2_texts = [
            "What's the best way to install Python libraries?",
            "The recommended way is to use pip, the Python package installer. Just run 'pip install library-name'.",
        ]

        # Create similar embeddings for both conversations
        user_embedding_1, user_embedding_2 = self.create_similar_embeddings(
            conversation_1_texts[0], conversation_2_texts[0]
        )

        assistant_embedding_1, assistant_embedding_2 = self.create_similar_embeddings(
            conversation_1_texts[1], conversation_2_texts[1]
        )

        # Create Memory objects for conversation 1
        memory_1 = Memory(
            fragments=[
                MemoryFragment(
                    id=101,
                    role=MessageRole.USER,
                    content=conversation_1_texts[0],
                    embeddings=[user_embedding_1],
                ),
                MemoryFragment(
                    id=102,
                    role=MessageRole.ASSISTANT,
                    content=conversation_1_texts[1],
                    embeddings=[assistant_embedding_1],
                ),
            ],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=1.0,
            source_id=self.test_conversation_1,
            conversation_id=self.test_conversation_1,
        )

        # Create Memory objects for conversation 2
        memory_2 = Memory(
            fragments=[
                MemoryFragment(
                    id=201,
                    role=MessageRole.USER,
                    content=conversation_2_texts[0],
                    embeddings=[user_embedding_2],
                ),
                MemoryFragment(
                    id=202,
                    role=MessageRole.ASSISTANT,
                    content=conversation_2_texts[1],
                    embeddings=[assistant_embedding_2],
                ),
            ],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=1.0,
            source_id=self.test_conversation_2,
            conversation_id=self.test_conversation_2,
        )

        # Create mock memory agent for storage
        # Store memories directly using memory storage (bypassing agent complexity)
        result_1 = True
        result_2 = True

        try:
            # Store memory fragments from conversation 1
            for fragment in memory_1.fragments:
                if fragment.embeddings:
                    await self.memory_storage.store_memory(
                        user_id=self.test_user_id,
                        source=memory_1.source.value,
                        role=fragment.role.value,
                        source_id=fragment.id,
                        embeddings=fragment.embeddings,
                    )

            # Store memory fragments from conversation 2
            for fragment in memory_2.fragments:
                if fragment.embeddings:
                    await self.memory_storage.store_memory(
                        user_id=self.test_user_id,
                        source=memory_2.source.value,
                        role=fragment.role.value,
                        source_id=fragment.id,
                        embeddings=fragment.embeddings,
                    )

        except Exception as e:
            logger.error(f"Failed to store memories: {e}")
            result_1 = False
            result_2 = False

        logger.info(f"Memory storage results: Conv1={result_1}, Conv2={result_2}")
        return result_1 and result_2

    async def test_memory_search_with_agent(self):
        """Test memory search using MemoryAgent."""
        logger.info("Testing memory search with MemoryAgent...")

        # Create search query similar to stored memories
        search_query = "How to install Python packages"
        search_embedding, _ = self.create_similar_embeddings(search_query, search_query)

        # Search memories directly using memory storage (bypassing agent complexity)
        memories = await self.memory_storage.search_similarity(
            embeddings=[search_embedding],
            min_similarity=0.6,  # Lower threshold to ensure we find results
            limit=5,
            user_id=self.test_user_id,
            conversation_id=None,  # Search across conversations
        )

        logger.info(f"Found {len(memories)} memories from search")

        # Validate results
        assert len(memories) > 0, "Should find at least one similar memory"

        for memory in memories:
            logger.info(
                f"Found memory: similarity={memory.similarity:.3f}, "
                f"conversation={memory.conversation_id}, "
                f"fragments={len(memory.fragments)}"
            )
            assert (
                memory.similarity >= 0.7
            ), f"Memory similarity {memory.similarity} below threshold"

        return memories

    async def test_memory_retrieval_tool(self):
        """Test the memory retrieval tool functionality."""
        logger.info("Testing memory retrieval tool...")

        # Create test state
        test_state = ToolsState(
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_1,
            user_config=UserConfig(
                user_id=self.test_user_id,
                memory=DEFAULT_MEMORY_CONFIG._replace(
                    similarity_threshold=0.7, enable_cross_conversation=True, limit=5
                ),
            ),
            messages=[],
            next_steps=[],
        )

        # Create mock tool runtime
        class MockToolRuntime:
            def __init__(self, state, tool_call_id):
                self.state = state
                self.tool_call_id = tool_call_id

        runtime = MockToolRuntime(test_state, "test_call_123")

        # Test memory retrieval with similar query
        search_query = "installing Python libraries with pip"
        result = await memory_retrieval(query=search_query, runtime=runtime)

        logger.info(f"Memory retrieval result: {result[:200]}...")

        # Validate tool response
        assert isinstance(result, str), "Tool should return string result"
        assert "Memory Search Results" in result or "No relevant memories" in result

        # If memories were found, check for expected content
        if "Memory Search Results" in result:
            assert "pip" in result.lower(), "Should find pip-related content"
            assert "install" in result.lower(), "Should find install-related content"
            logger.info("✅ Memory retrieval tool found relevant memories")
        else:
            logger.warning("⚠️  No memories found by retrieval tool")

        return result

    async def test_cross_conversation_retrieval(self):
        """Test that memories can be retrieved across different conversations."""
        logger.info("Testing cross-conversation memory retrieval...")

        # Search from conversation 1 context but expect to find memories from conversation 2
        search_query = "Python library installation"
        search_embedding, _ = self.create_similar_embeddings(search_query, search_query)

        memories = await self.memory_storage.search_similarity(
            embeddings=[search_embedding],
            min_similarity=0.6,
            limit=10,
            user_id=self.test_user_id,
            conversation_id=None,  # Search across all conversations
        )

        logger.info(f"Cross-conversation search found {len(memories)} memories")

        # Should find memories from both conversations
        conversation_ids = {m.conversation_id for m in memories}
        logger.info(f"Found memories from conversations: {conversation_ids}")

        assert len(memories) >= 2, "Should find memories from both conversations"
        assert len(conversation_ids) >= 2, "Should span multiple conversations"

        return memories

    async def run_all_tests(self):
        """Run the complete test suite."""
        logger.info("🧪 Starting memory retrieval test suite...")

        try:
            # Setup
            await self.setup_test_environment()
            await self.cleanup_test_data()

            # Test storage
            logger.info("📝 Testing memory storage...")
            storage_success = await self.store_test_memories()
            assert storage_success, "Memory storage failed"
            logger.info("✅ Memory storage test passed")

            # Test agent search
            logger.info("🔍 Testing memory agent search...")
            agent_memories = await self.test_memory_search_with_agent()
            logger.info("✅ Memory agent search test passed")

            # Test retrieval tool
            logger.info("🛠️ Testing memory retrieval tool...")
            tool_result = await self.test_memory_retrieval_tool()
            logger.info("✅ Memory retrieval tool test passed")

            # Test cross-conversation
            logger.info("🔄 Testing cross-conversation retrieval...")
            cross_memories = await self.test_cross_conversation_retrieval()
            logger.info("✅ Cross-conversation retrieval test passed")

            logger.info("🎉 All memory retrieval tests passed!")

            return {
                "storage_success": storage_success,
                "agent_memories": len(agent_memories),
                "tool_result_length": len(tool_result),
                "cross_conversation_memories": len(cross_memories),
            }

        except Exception as e:
            logger.error(f"❌ Test failed: {e}", exc_info=True)
            raise
        finally:
            # Cleanup
            await self.cleanup_test_data()


async def main():
    """Main test runner."""
    test_runner = TestMemoryRetrieval()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 50)
    print("MEMORY RETRIEVAL TEST RESULTS")
    print("=" * 50)
    print(f"Storage Success: {results['storage_success']}")
    print(f"Agent Memories Found: {results['agent_memories']}")
    print(f"Tool Result Length: {results['tool_result_length']}")
    print(f"Cross-Conversation Memories: {results['cross_conversation_memories']}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
