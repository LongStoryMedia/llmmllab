#!/usr/bin/env python3
"""
Test script to verify the dynamic_tools database schema matches DynamicTool model.

This is a debug script for manual verification on the remote cluster with real database.
"""

import asyncio
import sys
import logging
from models.dynamic_tool import DynamicTool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_dynamic_tools_schema():
    """Test that the database schema supports the DynamicTool model."""
    try:
        from db import storage  # pylint: disable=import-outside-toplevel

        if not storage.initialized:
            logger.error("❌ Database storage not initialized")
            return False

        # Test creating a tool with all BaseTool interface fields
        test_tool = DynamicTool(
            name="schema_test_tool",
            description="Test tool to validate database schema",
            code="def schema_test(): return 'schema validated'",
            function_name="schema_test",
            user_id="test_user_schema_validation",
            # BaseTool interface fields
            args_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            return_direct=True,
            verbose=True,
            tags=["test", "schema", "validation"],
            metadata={"test_type": "schema_validation", "version": "1.0"},
            handle_tool_error="Log and continue",
            handle_validation_error=False,
            response_format="content_and_artifact",
        )

        logger.info(
            "✅ DynamicTool model created successfully with all BaseTool fields"
        )

        # Test database operations
        created_tool = await storage.dynamic_tool.create_tool(test_tool)
        logger.info(f"✅ Tool created in database with ID: {created_tool.id}")

        # Test retrieval
        retrieved_tool = await storage.dynamic_tool.get_tool_by_id(
            created_tool.id, test_tool.user_id
        )

        if retrieved_tool:
            logger.info("✅ Tool retrieved from database successfully")

            # Verify BaseTool fields are preserved
            assert retrieved_tool.args_schema == test_tool.args_schema
            assert retrieved_tool.return_direct == test_tool.return_direct
            assert retrieved_tool.verbose == test_tool.verbose
            assert retrieved_tool.tags == test_tool.tags
            assert retrieved_tool.metadata == test_tool.metadata
            assert retrieved_tool.handle_tool_error == test_tool.handle_tool_error
            assert (
                retrieved_tool.handle_validation_error
                == test_tool.handle_validation_error
            )
            assert retrieved_tool.response_format == test_tool.response_format

            logger.info(
                "✅ All BaseTool interface fields preserved in database roundtrip"
            )

            # Test update
            retrieved_tool.description = "Updated description for schema test"
            retrieved_tool.tags = ["updated", "test"]
            retrieved_tool.metadata = {"updated": True}

            updated_tool = await storage.dynamic_tool.update_tool(retrieved_tool)
            logger.info("✅ Tool updated in database successfully")

            # Test deletion
            deleted = await storage.dynamic_tool.delete_tool(
                created_tool.id, test_tool.user_id
            )

            if deleted:
                logger.info("✅ Tool deleted from database successfully")
            else:
                logger.warning("⚠️ Tool deletion returned False")

        else:
            logger.error("❌ Failed to retrieve tool from database")
            return False

        logger.info("🎉 Database schema fully supports DynamicTool BaseTool interface!")
        return True

    except Exception as e:
        logger.error(f"❌ Schema test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    logger.info("🔍 Testing DynamicTool database schema compatibility...")

    success = await test_dynamic_tools_schema()

    if success:
        logger.info("✅ All database schema tests passed!")
        return 0
    else:
        logger.error("❌ Database schema tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
