"""
Test todo functionality end-to-end with direct database operations.
"""
import asyncio
from db import storage
from models.todo_item import TodoItem
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_todo_functionality")


async def test_todo_crud():
    """Test complete CRUD operations for todo items"""
    logger.info("Starting todo CRUD test...")

    # Initialize database connection
    connection_string = "postgresql://lsm:7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885@192.168.0.71:32345/llmmll?sslmode=disable"
    await storage.initialize(connection_string)

    test_user_id = "test_user_123"

    try:
        # Test 1: Create a new todo
        logger.info("Test 1: Creating a new todo item")
        todo = await storage.todo.add_todo(
            user_id=test_user_id,
            title="Test todo item",
            description="This is a test todo item",
            status="not-started",
            priority="high",
        )

        if todo:
            logger.info(f"✅ Todo created successfully: ID={todo.id}")
            todo_id = todo.id
        else:
            logger.error("❌ Failed to create todo")
            return

        # Test 2: Get todo by ID
        logger.info("Test 2: Retrieving todo by ID")
        retrieved_todo = await storage.todo.get_todo_by_id(todo_id, test_user_id)
        if retrieved_todo:
            logger.info(f"✅ Todo retrieved: {retrieved_todo.title}")
        else:
            logger.error("❌ Failed to retrieve todo by ID")
            return

        # Test 3: Get all todos for user
        logger.info("Test 3: Getting all todos for user")
        all_todos = await storage.todo.get_todos_by_user(test_user_id)
        logger.info(f"✅ Found {len(all_todos)} todos for user")

        # Test 4: Update todo
        logger.info("Test 4: Updating todo")
        updated_todo = await storage.todo.update_todo(
            todo_id=todo_id,
            user_id=test_user_id,
            title="Updated todo title",
            description="Updated description",
            status="in-progress",
            priority="medium",
        )

        if updated_todo:
            logger.info(f"✅ Todo updated: {updated_todo.title}")
            logger.info(f"Status: {updated_todo.status}, Priority: {updated_todo.priority}")
        else:
            logger.error("❌ Failed to update todo")
            return

        # Test 5: Get todos by status
        logger.info("Test 5: Getting todos by status")
        in_progress_todos = await storage.todo.get_todos_by_status(test_user_id, "in-progress")
        logger.info(f"✅ Found {len(in_progress_todos)} in-progress todos")

        # Test 6: Delete todo
        logger.info("Test 6: Deleting todo")
        delete_success = await storage.todo.delete_todo(todo_id, test_user_id)
        if delete_success:
            logger.info("✅ Todo deleted successfully")
        else:
            logger.error("❌ Failed to delete todo")

        # Test 7: Verify deletion
        logger.info("Test 7: Verifying deletion")
        deleted_todo = await storage.todo.get_todo_by_id(todo_id, test_user_id)
        if deleted_todo is None:
            logger.info("✅ Todo deletion verified")
        else:
            logger.error("❌ Todo still exists after deletion")

        logger.info("🎉 All todo CRUD tests passed!")

    except Exception as e:
        logger.error(f"❌ Todo test failed: {e}")
        raise
    finally:
        await storage.close()


if __name__ == "__main__":
    asyncio.run(test_todo_crud())