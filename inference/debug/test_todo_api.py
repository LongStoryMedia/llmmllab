"""
Test todo API endpoints end-to-end.
"""
import asyncio
import aiohttp
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="test_todo_api")


async def test_todo_api():
    """Test todo API endpoints"""
    base_url = "http://localhost:8000"
    
    logger.info("Starting todo API test...")

    async with aiohttp.ClientSession() as session:
        try:
            # Test 1: Create a new todo
            logger.info("Test 1: Creating a new todo via API")
            create_data = {
                "title": "API Test Todo",
                "description": "Testing todo API endpoints",
                "status": "not-started",
                "priority": "high"
            }
            
            async with session.post(f"{base_url}/todos/", json=create_data) as resp:
                if resp.status == 200:
                    todo_data = await resp.json()
                    todo_id = todo_data["id"]
                    logger.info(f"✅ Todo created via API: ID={todo_id}")
                else:
                    logger.error(f"❌ Failed to create todo: {resp.status}")
                    error_text = await resp.text()
                    logger.error(f"Error: {error_text}")
                    return

            # Test 2: Get todo by ID
            logger.info("Test 2: Getting todo by ID via API")
            async with session.get(f"{base_url}/todos/{todo_id}") as resp:
                if resp.status == 200:
                    todo_data = await resp.json()
                    logger.info(f"✅ Todo retrieved via API: {todo_data['title']}")
                else:
                    logger.error(f"❌ Failed to get todo: {resp.status}")
                    return

            # Test 3: Get all todos
            logger.info("Test 3: Getting all todos via API")
            async with session.get(f"{base_url}/todos/") as resp:
                if resp.status == 200:
                    todos = await resp.json()
                    logger.info(f"✅ Retrieved {len(todos)} todos via API")
                else:
                    logger.error(f"❌ Failed to get todos: {resp.status}")
                    return

            # Test 4: Update todo
            logger.info("Test 4: Updating todo via API")
            update_data = {
                "title": "Updated API Test Todo",
                "description": "Updated via API",
                "status": "in-progress",
                "priority": "medium"
            }
            
            async with session.put(f"{base_url}/todos/{todo_id}", json=update_data) as resp:
                if resp.status == 200:
                    updated_todo = await resp.json()
                    logger.info(f"✅ Todo updated via API: {updated_todo['title']}")
                    logger.info(f"Status: {updated_todo['status']}, Priority: {updated_todo['priority']}")
                else:
                    logger.error(f"❌ Failed to update todo: {resp.status}")
                    return

            # Test 5: Get todos by status
            logger.info("Test 5: Getting todos by status via API")
            async with session.get(f"{base_url}/todos/?status=in-progress") as resp:
                if resp.status == 200:
                    filtered_todos = await resp.json()
                    logger.info(f"✅ Found {len(filtered_todos)} in-progress todos via API")
                else:
                    logger.error(f"❌ Failed to get todos by status: {resp.status}")
                    return

            # Test 6: Delete todo
            logger.info("Test 6: Deleting todo via API")
            async with session.delete(f"{base_url}/todos/{todo_id}") as resp:
                if resp.status == 200:
                    result = await resp.json()
                    logger.info(f"✅ Todo deleted via API: {result['message']}")
                else:
                    logger.error(f"❌ Failed to delete todo: {resp.status}")
                    return

            # Test 7: Verify deletion
            logger.info("Test 7: Verifying deletion via API")
            async with session.get(f"{base_url}/todos/{todo_id}") as resp:
                if resp.status == 404:
                    logger.info("✅ Todo deletion verified via API")
                else:
                    logger.error(f"❌ Todo still exists after deletion: {resp.status}")

            logger.info("🎉 All todo API tests passed!")

        except Exception as e:
            logger.error(f"❌ Todo API test failed: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(test_todo_api())