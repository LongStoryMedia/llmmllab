#!/usr/bin/env python3
"""
Test script for conversation-linked todos functionality.
Run this in the container to verify the todo system works with conversation linking.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/app')

from db import storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="todo_test")

async def test_conversation_todos():
    """Test conversation-linked todo functionality."""
    
    if not storage.initialized:
        logger.error("Database not initialized")
        return False
    
    if not storage.todo:
        logger.error("Todo storage not initialized")
        return False
    
    test_user_id = "test_user_123"
    test_conversation_id = 42
    
    try:
        # Test 1: Create a todo linked to a conversation
        logger.info("Creating conversation-linked todo...")
        todo = await storage.todo.add_todo(
            user_id=test_user_id,
            title="Test conversation todo",
            description="This todo is linked to a specific conversation",
            status="not-started",
            priority="medium",
            conversation_id=test_conversation_id
        )
        
        if not todo:
            logger.error("Failed to create conversation-linked todo")
            return False
        
        logger.info(f"Created todo with ID {todo.id}, conversation_id: {todo.conversation_id}")
        
        # Test 2: Create a todo without conversation link
        logger.info("Creating regular todo...")
        regular_todo = await storage.todo.add_todo(
            user_id=test_user_id,
            title="Regular todo",
            description="This todo has no conversation link",
            status="not-started",
            priority="high"
        )
        
        if not regular_todo:
            logger.error("Failed to create regular todo")
            return False
        
        logger.info(f"Created regular todo with ID {regular_todo.id}, conversation_id: {regular_todo.conversation_id}")
        
        # Test 3: Get todos by conversation
        logger.info(f"Fetching todos for conversation {test_conversation_id}...")
        conversation_todos = await storage.todo.get_todos_by_conversation(
            user_id=test_user_id, 
            conversation_id=test_conversation_id
        )
        
        logger.info(f"Found {len(conversation_todos)} todos for conversation {test_conversation_id}")
        for todo in conversation_todos:
            logger.info(f"  - Todo {todo.id}: {todo.title} (conversation_id: {todo.conversation_id})")
        
        # Test 4: Get all todos for user
        logger.info("Fetching all todos for user...")
        all_todos = await storage.todo.get_todos_by_user(test_user_id)
        
        logger.info(f"Found {len(all_todos)} total todos for user")
        for todo in all_todos:
            logger.info(f"  - Todo {todo.id}: {todo.title} (conversation_id: {todo.conversation_id})")
        
        # Test 5: Clean up - delete test todos
        logger.info("Cleaning up test todos...")
        await storage.todo.delete_todo(todo.id, test_user_id)
        await storage.todo.delete_todo(regular_todo.id, test_user_id)
        
        logger.info("✅ All conversation todo tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting conversation todo tests...")
    
    # Initialize database connection if needed
    if not storage.initialized:
        logger.info("Initializing database connection...")
        connection_string = os.getenv('DATABASE_URL', 'postgresql://lsm:password@psql-service.psql.svc.cluster.local:5432/llmmll')
        await storage.initialize(connection_string)
    
    success = await test_conversation_todos()
    
    if success:
        logger.info("🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())