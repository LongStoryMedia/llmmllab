"""
Storage service for managing todo items in the database.
Todo items represent user task management with priority and status tracking.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime
from models.todo_item import TodoItem
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="todo_storage")


class TodoStorage:
    """Storage service for todo items with CRUD operations."""

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="todo_storage_instance")

    async def add_todo(self, todo_item: TodoItem) -> Optional[TodoItem]:
        """
        Add a new todo item to the database.

        Args:
            todo_item: The TodoItem object to add to the database

        Returns:
            The created TodoItem with database-generated fields, or None on failure
        """
        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    self.get_query("todo.add_todo"),
                    todo_item.user_id,
                    todo_item.conversation_id,
                    todo_item.title,
                    todo_item.description,
                    todo_item.status,
                    todo_item.priority,
                    todo_item.due_date,
                )

                if row:
                    # Return a new TodoItem with the database-generated fields
                    return TodoItem(
                        id=row["id"],
                        user_id=todo_item.user_id,
                        conversation_id=todo_item.conversation_id,
                        title=todo_item.title,
                        description=todo_item.description,
                        status=todo_item.status,
                        priority=todo_item.priority,
                        due_date=todo_item.due_date,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                return None

        except Exception as e:
            self.logger.error(f"Failed to add todo: {e}")
            return None

    async def get_todos_by_user(self, user_id: str) -> List[TodoItem]:
        """
        Get all todos for a specific user, ordered by priority and creation date.

        Args:
            user_id: ID of the user

        Returns:
            List of TodoItem objects
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(self.get_query("todo.get_by_user"), user_id)

                return [
                    TodoItem(
                        id=row["id"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        title=row["title"],
                        description=row["description"],
                        status=row["status"],
                        priority=row["priority"],
                        due_date=row["due_date"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            self.logger.error(f"Failed to get todos for user {user_id}: {e}")
            return []

    async def get_todo_by_id(self, todo_id: int, user_id: str) -> Optional[TodoItem]:
        """
        Get a specific todo by ID and user_id.

        Args:
            todo_id: ID of the todo item
            user_id: ID of the user (for ownership verification)

        Returns:
            TodoItem if found, None otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    self.get_query("todo.get_by_id"), todo_id, user_id
                )

                if row:
                    return TodoItem(
                        id=row["id"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        title=row["title"],
                        description=row["description"],
                        status=row["status"],
                        priority=row["priority"],
                        due_date=row["due_date"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                return None

        except Exception as e:
            self.logger.error(f"Failed to get todo {todo_id}: {e}")
            return None

    async def update_todo(self, todo_item: TodoItem) -> Optional[TodoItem]:
        """
        Update an existing todo item.

        Args:
            todo_item: The TodoItem object with updated values (must have id set)

        Returns:
            Updated TodoItem if successful, None otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    self.get_query("todo.update_todo"),
                    todo_item.id,
                    todo_item.user_id,
                    todo_item.title,
                    todo_item.description,
                    todo_item.status,
                    todo_item.priority,
                    todo_item.due_date,
                )

                if row:
                    return TodoItem(
                        id=row["id"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        title=row["title"],
                        description=row["description"],
                        status=row["status"],
                        priority=row["priority"],
                        due_date=row["due_date"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                return None

        except Exception as e:
            self.logger.error(f"Failed to update todo {todo_item.id}: {e}")
            return None

    async def delete_todo(self, todo_id: int, user_id: str) -> bool:
        """
        Delete a todo item.

        Args:
            todo_id: ID of the todo to delete
            user_id: ID of the user (for ownership verification)

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    self.get_query("todo.delete_todo"), todo_id, user_id
                )
                return row is not None

        except Exception as e:
            self.logger.error(f"Failed to delete todo {todo_id}: {e}")
            return False

    async def get_todos_by_status(self, user_id: str, status: str) -> List[TodoItem]:
        """
        Get todos filtered by status for a specific user.

        Args:
            user_id: ID of the user
            status: Status to filter by

        Returns:
            List of TodoItem objects with the specified status
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("todo.get_by_status"), user_id, status
                )

                return [
                    TodoItem(
                        id=row["id"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        title=row["title"],
                        description=row["description"],
                        status=row["status"],
                        priority=row["priority"],
                        due_date=row["due_date"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            self.logger.error(
                f"Failed to get todos with status {status} for user {user_id}: {e}"
            )
            return []

    async def get_todos_by_conversation(
        self, user_id: str, conversation_id: int
    ) -> List[TodoItem]:
        """
        Get todos for a specific conversation and user.

        Args:
            user_id: ID of the user
            conversation_id: ID of the conversation

        Returns:
            List of TodoItem objects associated with the conversation
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("todo.get_by_conversation"), user_id, conversation_id
                )

                return [
                    TodoItem(
                        id=row["id"],
                        user_id=row["user_id"],
                        conversation_id=row["conversation_id"],
                        title=row["title"],
                        description=row["description"],
                        status=row["status"],
                        priority=row["priority"],
                        due_date=row["due_date"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                    )
                    for row in rows
                ]

        except Exception as e:
            self.logger.error(
                f"Failed to get todos for conversation {conversation_id} and user {user_id}: {e}"
            )
            return []
