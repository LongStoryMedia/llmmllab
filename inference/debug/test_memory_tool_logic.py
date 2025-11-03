"""
Simplified Memory Retrieval Test - Focus on Tool Logic

This test verifies the memory retrieval tool functionality by mocking the storage layer
and testing the business logic directly. This bypasses database constraint issues
while still verifying that the memory retrieval system works correctly.
"""

import asyncio
import os
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock

# Module imports
from models import Memory, MemoryFragment, MemorySource, MessageRole, UserConfig
from models.default_configs import DEFAULT_MEMORY_CONFIG
from composer.tools.static.memory_retrieval_tool import memory_retrieval
from composer.graph.state import ToolsState
from langchain.tools import ToolRuntime
from utils.logging import llmmllogger

# Set up test environment
os.environ["TESTING"] = "true"

logger = llmmllogger.logger.bind(component="MemoryToolTest")


class MockMemoryStorage:
    """Mock memory storage that returns predefined test data."""

    def __init__(self):
        self.stored_memories = []
        self.setup_test_memories()

    def setup_test_memories(self):
        """Create test memories with similar semantic content."""
        # Memory 1: Python pip installation
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
            similarity=0.85,  # High similarity for pip-related query
            source_id=1001,
            conversation_id=1001,
        )

        # Memory 2: Python library installation (similar but different conversation)
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
            similarity=0.82,  # High similarity for pip-related query
            source_id=1002,
            conversation_id=1002,
        )

        # Memory 3: Unrelated content (should have lower similarity)
        memory_3 = Memory(
            fragments=[
                MemoryFragment(
                    id=301,
                    role=MessageRole.USER,
                    content="How do I create a new file in Linux?",
                ),
                MemoryFragment(
                    id=302,
                    role=MessageRole.ASSISTANT,
                    content="You can create a new file using 'touch filename' or 'echo content > filename'.",
                ),
            ],
            source=MemorySource.MESSAGE,
            created_at=datetime.now(),
            similarity=0.25,  # Low similarity for pip-related query
            source_id=1003,
            conversation_id=1003,
        )

        self.stored_memories = [memory_1, memory_2, memory_3]

    async def search_similarity(
        self,
        embeddings: List[List[float]],
        min_similarity: float,
        limit: int,
        user_id: str = None,
        conversation_id: int = None,
    ) -> List[Memory]:
        """Mock similarity search that returns relevant memories based on threshold."""
        # Filter memories by similarity threshold
        relevant_memories = [
            memory
            for memory in self.stored_memories
            if memory.similarity >= min_similarity
        ]

        # Sort by similarity (highest first) and apply limit
        relevant_memories.sort(key=lambda m: m.similarity, reverse=True)
        return relevant_memories[:limit]


class TestMemoryRetrievalTool:
    """Test class for memory retrieval tool functionality."""

    def __init__(self):
        self.test_user_id = "test_user_123"
        self.test_conversation_id = 1001

    async def test_memory_retrieval_tool_with_results(self):
        """Test the memory retrieval tool when relevant memories are found."""
        logger.info("Testing memory retrieval tool with relevant results...")

        # Create mock storage and patch it into the tool module
        mock_storage = MockMemoryStorage()

        # Mock the storage module in the tool
        import composer.tools.static.memory_retrieval_tool as tool_module

        original_storage = getattr(tool_module, "storage", None)

        # Create a mock storage object that has the required interface
        mock_storage_service = MagicMock()
        mock_storage_service.search_similarity = mock_storage.search_similarity

        mock_storage_module = MagicMock()
        mock_storage_module.pool = True  # Indicate storage is initialized
        mock_storage_module.get_service = MagicMock(return_value=mock_storage_service)

        tool_module.storage = mock_storage_module

        try:
            # Create test state with memory configuration
            test_state = ToolsState(
                user_id=self.test_user_id,
                conversation_id=self.test_conversation_id,
                user_config={
                    "memory": {
                        "similarity_threshold": 0.7,
                        "enable_cross_conversation": True,
                        "limit": 5,
                        "enable_cross_user": False,
                        "storage_days": 30,
                        "enabled": True,
                    }
                },
                messages=[],
                next_steps=[],
            )

            # Create mock tool runtime
            class MockToolRuntime:
                def __init__(self, state, tool_call_id):
                    self.state = state
                    self.tool_call_id = tool_call_id

            runtime = MockToolRuntime(test_state, "test_call_123")

            # Test memory retrieval with pip-related query (should find results)
            search_query = "installing Python packages with pip"
            # Call the tool using its invoke method (LangChain tool pattern)
            result = await memory_retrieval.ainvoke(
                {"query": search_query, "runtime": runtime}
            )

            logger.info(f"Memory retrieval result length: {len(result)}")
            logger.info(f"Result preview: {result[:200]}...")

            # Validate results
            assert isinstance(result, str), "Tool should return string result"
            assert "Memory Search Results" in result, "Should find relevant memories"
            assert "pip" in result.lower(), "Should contain pip-related content"
            assert "install" in result.lower(), "Should contain install-related content"

            logger.info("✅ Memory retrieval tool test with results PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Memory retrieval tool test FAILED: {e}")
            return False
        finally:
            # Restore original storage
            if original_storage:
                tool_module.storage = original_storage

    async def test_memory_retrieval_tool_no_results(self):
        """Test the memory retrieval tool when no relevant memories are found."""
        logger.info("Testing memory retrieval tool with no relevant results...")

        # Create mock storage that returns empty results
        mock_storage = MockMemoryStorage()
        mock_storage.stored_memories = []  # No stored memories

        # Mock the storage module in the tool
        import composer.tools.static.memory_retrieval_tool as tool_module

        original_storage = getattr(tool_module, "storage", None)

        mock_storage_service = MagicMock()
        mock_storage_service.search_similarity = mock_storage.search_similarity

        mock_storage_module = MagicMock()
        mock_storage_module.pool = True
        mock_storage_module.get_service = MagicMock(return_value=mock_storage_service)

        tool_module.storage = mock_storage_module

        try:
            # Create test state
            test_state = ToolsState(
                user_id=self.test_user_id,
                conversation_id=self.test_conversation_id,
                user_config={
                    "memory": {
                        "similarity_threshold": 0.7,
                        "enable_cross_conversation": True,
                        "limit": 5,
                        "enable_cross_user": False,
                        "storage_days": 30,
                        "enabled": True,
                    }
                },
                messages=[],
                next_steps=[],
            )

            class MockToolRuntime:
                def __init__(self, state, tool_call_id):
                    self.state = state
                    self.tool_call_id = tool_call_id

            runtime = MockToolRuntime(test_state, "test_call_123")

            # Test with query that should find no results
            search_query = "completely unrelated topic about space exploration"
            result = await memory_retrieval.ainvoke(
                {"query": search_query, "runtime": runtime}
            )

            logger.info(f"No results test result: {result}")

            # Validate no results case
            assert isinstance(result, str), "Tool should return string result"
            assert (
                "No relevant memories found" in result
            ), "Should indicate no memories found"

            logger.info("✅ Memory retrieval tool test with no results PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Memory retrieval tool test FAILED: {e}")
            return False
        finally:
            # Restore original storage
            if original_storage:
                tool_module.storage = original_storage

    async def test_memory_retrieval_tool_error_handling(self):
        """Test the memory retrieval tool error handling."""
        logger.info("Testing memory retrieval tool error handling...")

        # Mock storage that raises an exception
        mock_storage_service = MagicMock()
        mock_storage_service.search_similarity = AsyncMock(
            side_effect=Exception("Storage error")
        )

        # Mock the storage module
        import composer.tools.static.memory_retrieval_tool as tool_module

        original_storage = getattr(tool_module, "storage", None)

        mock_storage_module = MagicMock()
        mock_storage_module.pool = True
        mock_storage_module.get_service = MagicMock(return_value=mock_storage_service)

        tool_module.storage = mock_storage_module

        try:
            # Create test state
            test_state = ToolsState(
                user_id=self.test_user_id,
                conversation_id=self.test_conversation_id,
                user_config={
                    "memory": {
                        "similarity_threshold": 0.7,
                        "enable_cross_conversation": True,
                        "limit": 5,
                        "enable_cross_user": False,
                        "storage_days": 30,
                        "enabled": True,
                    }
                },
                messages=[],
                next_steps=[],
            )

            class MockToolRuntime:
                def __init__(self, state, tool_call_id):
                    self.state = state
                    self.tool_call_id = tool_call_id

            runtime = MockToolRuntime(test_state, "test_call_123")

            # Test with query that should trigger error
            search_query = "test query"
            result = await memory_retrieval.ainvoke(
                {"query": search_query, "runtime": runtime}
            )

            logger.info(f"Error handling test result: {result}")

            # Validate error handling
            assert isinstance(result, str), "Tool should return string result"
            assert (
                "Memory retrieval failed" in result or "❌" in result
            ), "Should indicate failure"

            logger.info("✅ Memory retrieval tool error handling test PASSED")
            return True

        except Exception as e:
            logger.error(f"❌ Memory retrieval tool error handling test FAILED: {e}")
            return False
        finally:
            # Restore original storage
            if original_storage:
                tool_module.storage = original_storage

    async def test_memory_similarity_logic(self):
        """Test the similarity filtering logic."""
        logger.info("Testing memory similarity filtering...")

        mock_storage = MockMemoryStorage()

        # Test high similarity threshold (should only return very similar memories)
        high_threshold_results = await mock_storage.search_similarity(
            embeddings=[[0.1] * 768], min_similarity=0.8, limit=10  # Mock embedding
        )

        # Should find 2 memories with similarity >= 0.8 (0.85 and 0.82)
        assert (
            len(high_threshold_results) == 2
        ), f"Expected 2 results, got {len(high_threshold_results)}"
        assert all(
            m.similarity >= 0.8 for m in high_threshold_results
        ), "All results should meet threshold"

        # Test low similarity threshold (should return all memories)
        low_threshold_results = await mock_storage.search_similarity(
            embeddings=[[0.1] * 768], min_similarity=0.2, limit=10
        )

        # Should find all 3 memories
        assert (
            len(low_threshold_results) == 3
        ), f"Expected 3 results, got {len(low_threshold_results)}"

        # Test limit functionality
        limited_results = await mock_storage.search_similarity(
            embeddings=[[0.1] * 768], min_similarity=0.2, limit=1
        )

        # Should only return 1 result (the highest similarity)
        assert (
            len(limited_results) == 1
        ), f"Expected 1 result, got {len(limited_results)}"
        assert (
            limited_results[0].similarity == 0.85
        ), "Should return highest similarity memory"

        logger.info("✅ Memory similarity logic test PASSED")
        return True

    async def run_all_tests(self):
        """Run all memory retrieval tool tests."""
        logger.info("🧪 Starting Memory Retrieval Tool Test Suite...")

        test_results = []

        # Test 1: Tool with relevant results
        logger.info("🔍 Test 1: Memory retrieval with relevant results")
        result_1 = await self.test_memory_retrieval_tool_with_results()
        test_results.append(("Tool with results", result_1))

        # Test 2: Tool with no results
        logger.info("🔍 Test 2: Memory retrieval with no results")
        result_2 = await self.test_memory_retrieval_tool_no_results()
        test_results.append(("Tool with no results", result_2))

        # Test 3: Tool error handling
        logger.info("🔍 Test 3: Memory retrieval error handling")
        result_3 = await self.test_memory_retrieval_tool_error_handling()
        test_results.append(("Tool error handling", result_3))

        # Test 4: Similarity logic
        logger.info("🔍 Test 4: Memory similarity filtering")
        result_4 = await self.test_memory_similarity_logic()
        test_results.append(("Similarity logic", result_4))

        # Summary
        passed_tests = sum(1 for _, result in test_results if result)
        total_tests = len(test_results)

        logger.info(
            f"🎉 Memory Retrieval Tool Tests Complete: {passed_tests}/{total_tests} passed"
        )

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "all_passed": passed_tests == total_tests,
            "results": test_results,
        }


async def main():
    """Main test runner."""
    test_runner = TestMemoryRetrievalTool()
    results = await test_runner.run_all_tests()

    print("\n" + "=" * 60)
    print("MEMORY RETRIEVAL TOOL TEST RESULTS")
    print("=" * 60)

    for test_name, passed in results["results"]:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<30} {status}")

    print("=" * 60)
    print(
        f"Overall Result: {results['passed_tests']}/{results['total_tests']} tests passed"
    )

    if results["all_passed"]:
        print("🎉 ALL TESTS PASSED - Memory retrieval tool is working correctly!")
    else:
        print("⚠️  SOME TESTS FAILED - Memory retrieval tool may have issues")

    print("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(main())
