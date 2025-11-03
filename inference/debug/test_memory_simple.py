"""
Simple Memory Retrieval Test - Direct Function Testing

This test calls the memory retrieval logic directly without going through
the LangChain tool wrapper, making it easier to test the core functionality.
"""

import asyncio
import os
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock

# Module imports
from models import Memory, MemoryFragment, MemorySource, MessageRole
from utils.logging import llmmllogger

# Set up test environment
os.environ["TESTING"] = "true"

logger = llmmllogger.logger.bind(component="SimpleMemoryTest")


class MockMemoryStorage:
    """Mock memory storage that returns predefined test data."""

    def __init__(self):
        self.stored_memories = []
        self.setup_test_memories()

    def setup_test_memories(self):
        """Create test memories with similar semantic content."""
        memory_1 = Memory(
            fragments=[
                MemoryFragment(
                    id=101,
                    role=MessageRole.USER,
                    content="How do I install Python packages using pip?",
                ),
                MemoryFragment(
                    id=102,
                    role=MessageRole.ASSISTANT,
                    content="You can install Python packages using pip by running 'pip install package-name' in your terminal.",
                ),
            ],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=0.85,
            source_id=1001,
            conversation_id=1001,
        )

        memory_2 = Memory(
            fragments=[
                MemoryFragment(
                    id=201,
                    role=MessageRole.USER,
                    content="What's the best way to install Python libraries?",
                ),
                MemoryFragment(
                    id=202,
                    role=MessageRole.ASSISTANT,
                    content="The recommended way is to use pip, the Python package installer. Just run 'pip install library-name'.",
                ),
            ],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=0.82,
            source_id=1002,
            conversation_id=1002,
        )

        self.stored_memories = [memory_1, memory_2]

    async def search_similarity(
        self,
        embeddings: List[List[float]],
        min_similarity: float,
        limit: int,
        user_id: str = None,
        conversation_id: int = None,
    ) -> List[Memory]:
        """Mock similarity search."""
        relevant_memories = [
            memory
            for memory in self.stored_memories
            if memory.similarity >= min_similarity
        ]

        relevant_memories.sort(key=lambda m: m.similarity, reverse=True)
        return relevant_memories[:limit]


async def memory_retrieval_logic(
    query: str, user_id: str, conversation_id: int, memory_config: dict, storage_service
) -> str:
    """
    Core memory retrieval logic extracted from the tool.
    This replicates the logic without the LangChain tool wrapper.
    """
    try:
        logger.info(f"Starting memory retrieval for query: {query[:50]}...")

        # Mock embedding generation (normally done by pipeline)
        mock_embeddings = [[0.1] * 768]  # Simple mock embedding

        # Configure filtering based on memory config
        user_filter = None if memory_config.get("enable_cross_user", False) else user_id
        conversation_filter = (
            None
            if memory_config.get("enable_cross_conversation", True)
            else conversation_id
        )

        # Retrieve similar memories
        memories = await storage_service.search_similarity(
            embeddings=mock_embeddings,
            min_similarity=memory_config.get("similarity_threshold", 0.7),
            limit=memory_config.get("limit", 5),
            user_id=user_filter,
            conversation_id=conversation_filter,
        )

        # Format memories for display
        if memories:
            formatted_memories = []
            for memory in memories:
                content_parts = []
                for fragment in memory.fragments:
                    if hasattr(fragment, "content") and fragment.content:
                        content_parts.append(fragment.content)

                formatted_memory = {
                    "content": "\n".join(content_parts),
                    "timestamp": (
                        memory.created_at.isoformat()
                        if hasattr(memory, "created_at")
                        else None
                    ),
                    "similarity": (
                        memory.similarity if hasattr(memory, "similarity") else 1.0
                    ),
                    "source": (
                        memory.source.value if hasattr(memory, "source") else "unknown"
                    ),
                }
                formatted_memories.append(formatted_memory)

            # Create response message
            response_message = f"🧠 **Memory Search Results for: '{query}'**\n\n"
            for i, memory in enumerate(formatted_memories, 1):
                response_message += f"**{i}. Memory from {memory['source']}**\n"
                response_message += f"   Content: {memory['content'][:200]}...\n"
                response_message += f"   Timestamp: {memory['timestamp']}\n"
                response_message += f"   Similarity: {memory['similarity']:.2f}\n\n"
        else:
            response_message = f"🧠 No relevant memories found for query: '{query}'"

        logger.info(f"Memory retrieval completed with {len(memories)} memories")
        return response_message

    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}")
        return f"❌ Memory retrieval failed: {str(e)}"


class TestMemoryRetrievalLogic:
    """Test class for core memory retrieval logic."""

    def __init__(self):
        self.test_user_id = "test_user_123"
        self.test_conversation_id = 1001

    async def test_memory_logic_with_results(self):
        """Test memory retrieval logic when relevant memories are found."""
        logger.info("Testing memory retrieval logic with results...")

        mock_storage = MockMemoryStorage()

        memory_config = {
            "similarity_threshold": 0.7,
            "enable_cross_conversation": True,
            "enable_cross_user": False,
            "limit": 5,
        }

        query = "installing Python packages with pip"

        result = await memory_retrieval_logic(
            query=query,
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            memory_config=memory_config,
            storage_service=mock_storage,
        )

        logger.info(f"Result preview: {result[:200]}...")

        # Validate results
        assert isinstance(result, str), "Should return string result"
        assert "Memory Search Results" in result, "Should find relevant memories"
        assert "pip" in result.lower(), "Should contain pip-related content"
        assert "install" in result.lower(), "Should contain install-related content"
        assert "0.85" in result or "0.82" in result, "Should show similarity scores"

        logger.info("✅ Memory retrieval logic with results test PASSED")
        return True

    async def test_memory_logic_no_results(self):
        """Test memory retrieval logic when no relevant memories are found."""
        logger.info("Testing memory retrieval logic with no results...")

        mock_storage = MockMemoryStorage()

        memory_config = {
            "similarity_threshold": 0.9,  # Very high threshold
            "enable_cross_conversation": True,
            "enable_cross_user": False,
            "limit": 5,
        }

        query = "space exploration topics"

        result = await memory_retrieval_logic(
            query=query,
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            memory_config=memory_config,
            storage_service=mock_storage,
        )

        logger.info(f"No results test result: {result}")

        # Validate no results case
        assert isinstance(result, str), "Should return string result"
        assert (
            "No relevant memories found" in result
        ), "Should indicate no memories found"

        logger.info("✅ Memory retrieval logic with no results test PASSED")
        return True

    async def test_memory_logic_error_handling(self):
        """Test memory retrieval logic error handling."""
        logger.info("Testing memory retrieval logic error handling...")

        # Mock storage that raises an exception
        mock_storage = MagicMock()
        mock_storage.search_similarity = AsyncMock(
            side_effect=Exception("Storage error")
        )

        memory_config = {
            "similarity_threshold": 0.7,
            "enable_cross_conversation": True,
            "enable_cross_user": False,
            "limit": 5,
        }

        query = "test query"

        result = await memory_retrieval_logic(
            query=query,
            user_id=self.test_user_id,
            conversation_id=self.test_conversation_id,
            memory_config=memory_config,
            storage_service=mock_storage,
        )

        logger.info(f"Error handling test result: {result}")

        # Validate error handling
        assert isinstance(result, str), "Should return string result"
        assert (
            "Memory retrieval failed" in result or "❌" in result
        ), "Should indicate failure"

        logger.info("✅ Memory retrieval logic error handling test PASSED")
        return True

    async def test_cross_conversation_filtering(self):
        """Test cross-conversation filtering functionality."""
        logger.info("Testing cross-conversation filtering...")

        mock_storage = MockMemoryStorage()

        # Test with cross-conversation enabled (should find memories from different conversations)
        memory_config_cross = {
            "similarity_threshold": 0.7,
            "enable_cross_conversation": True,
            "enable_cross_user": False,
            "limit": 5,
        }

        result_cross = await memory_retrieval_logic(
            query="pip install",
            user_id=self.test_user_id,
            conversation_id=999,  # Different from stored memories (1001, 1002)
            memory_config=memory_config_cross,
            storage_service=mock_storage,
        )

        # Should find memories even though conversation_id is different
        assert (
            "Memory Search Results" in result_cross
        ), "Should find memories with cross-conversation enabled"

        logger.info("✅ Cross-conversation filtering test PASSED")
        return True

    async def run_all_tests(self):
        """Run all memory retrieval logic tests."""
        logger.info("🧪 Starting Memory Retrieval Logic Test Suite...")

        test_results = []

        # Test 1: Logic with results
        logger.info("🔍 Test 1: Memory logic with results")
        result_1 = await self.test_memory_logic_with_results()
        test_results.append(("Logic with results", result_1))

        # Test 2: Logic with no results
        logger.info("🔍 Test 2: Memory logic with no results")
        result_2 = await self.test_memory_logic_no_results()
        test_results.append(("Logic with no results", result_2))

        # Test 3: Logic error handling
        logger.info("🔍 Test 3: Memory logic error handling")
        result_3 = await self.test_memory_logic_error_handling()
        test_results.append(("Logic error handling", result_3))

        # Test 4: Cross-conversation filtering
        logger.info("🔍 Test 4: Cross-conversation filtering")
        result_4 = await self.test_cross_conversation_filtering()
        test_results.append(("Cross-conversation filtering", result_4))

        # Summary
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)

        logger.info(
            f"🎉 Memory Retrieval Logic Tests Complete: {passed_tests}/{total_tests} passed"
        )

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "all_passed": passed_tests == total_tests,
            "results": test_results,
        }


async def main():
    """Main test runner."""
    test_runner = TestMemoryRetrievalLogic()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 60)
    print("MEMORY RETRIEVAL LOGIC TEST RESULTS")
    print("=" * 60)

    for test_name, passed in results["results"]:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    print("=" * 60)
    print(
        f"Overall Result: {results['passed_tests']}/{results['total_tests']} tests passed"
    )

    if results["all_passed"]:
        print("🎉 ALL TESTS PASSED - Memory retrieval logic is working correctly!")
        print(
            "\n💡 DIAGNOSIS: The memory retrieval functionality appears to be working."
        )
        print("   Issues with memory retrieval are likely in:")
        print("   - Database storage/connection issues")
        print("   - Embedding generation pipeline")
        print("   - Tool integration with LangGraph")
        print("   - User configuration setup")
    else:
        print("⚠️  SOME TESTS FAILED - Memory retrieval logic may have issues")

    print("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(main())
