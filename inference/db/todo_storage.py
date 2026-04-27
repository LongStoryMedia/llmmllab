"""
Storage service for managing todo items in the database.
Todo items represent user task management with priority and status tracking.
"""

from typing import List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.todo_item import TodoItem
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="todo_storage")

_SELECT_TODO_COLS = (
    "id, user_id, conversation_id, title, description, "
    "status, priority, due_date, created_at, updated_at"
)

_ORDER_BY_PRIORITY = (
    "CASE priority "
    "WHEN 'urgent' THEN 1 "
    "WHEN 'high' THEN 2 "
    "WHEN 'medium' THEN 3 "
    "WHEN 'low' THEN 4 "
    "END, created_at DESC"
)


class TodoStorage:
    """Storage service for todo items with CRUD operations."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self.session_factory = session_factory
        self.logger = llmmllogger.bind(component="todo_storage_instance")

    def _build_todo(self, row: dict) -> TodoItem:
        """Build a TodoItem from a row dict."""
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

    async def add_todo(self, todo_item: TodoItem) -> Optional[TodoItem]:
        """Add a new todo item to the database."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text("""
                        INSERT INTO todos(
                            user_id, conversation_id, title, description,
                            status, priority, due_date
                        ) VALUES (
                            :user_id, :conversation_id, :title, :description,
                            :status, :priority, :due_date
                        )
                        RETURNING id, created_at, updated_at
                    """),
                    {
                        "user_id": todo_item.user_id,
                        "conversation_id": todo_item.conversation_id,
                        "title": todo_item.title,
                        "description": todo_item.description,
                        "status": todo_item.status,
                        "priority": todo_item.priority,
                        "due_date": todo_item.due_date,
                    },
                )
                await session.commit()

                row = result.mappings().first()
                if row:
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
        """Get all todos for a specific user, ordered by priority and creation date."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(f"""
                        SELECT {_SELECT_TODO_COLS}
                        FROM todos
                        WHERE user_id = :user_id
                        ORDER BY {_ORDER_BY_PRIORITY}
                    """),
                    {"user_id": user_id},
                )
                return [self._build_todo(dict(row)) for row in result.mappings()]

        except Exception as e:
            self.logger.error(f"Failed to get todos for user {user_id}: {e}")
            return []

    async def get_todo_by_id(
        self, todo_id: int, user_id: str
    ) -> Optional[TodoItem]:
        """Get a specific todo by ID, ensuring ownership."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(f"""
                        SELECT {_SELECT_TODO_COLS}
                        FROM todos
                        WHERE id = :todo_id AND user_id = :user_id
                    """),
                    {"todo_id": todo_id, "user_id": user_id},
                )
                row = result.mappings().first()
                return self._build_todo(dict(row)) if row else None

        except Exception as e:
            self.logger.error(f"Failed to get todo {todo_id}: {e}")
            return None

    async def update_todo(self, todo_item: TodoItem) -> Optional[TodoItem]:
        """Update an existing todo item."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(f"""
                        UPDATE todos
                        SET title = :title,
                            description = :description,
                            status = :status,
                            priority = :priority,
                            due_date = :due_date
                        WHERE id = :todo_id AND user_id = :user_id
                        RETURNING {_SELECT_TODO_COLS}
                    """),
                    {
                        "todo_id": todo_item.id,  # type: ignore[arg-type]
                        "user_id": todo_item.user_id,
                        "title": todo_item.title,
                        "description": todo_item.description,
                        "status": todo_item.status,
                        "priority": todo_item.priority,
                        "due_date": todo_item.due_date,
                    },
                )
                await session.commit()
                row = result.mappings().first()
                return self._build_todo(dict(row)) if row else None

        except Exception as e:
            self.logger.error(f"Failed to update todo {todo_item.id}: {e}")  # type: ignore[arg-type]
            return None

    async def delete_todo(self, todo_id: int, user_id: str) -> bool:
        """Delete a todo by ID, ensuring ownership."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text("""
                        DELETE FROM todos
                        WHERE id = :todo_id AND user_id = :user_id
                        RETURNING id
                    """),
                    {"todo_id": todo_id, "user_id": user_id},
                )
                await session.commit()
                return result.mappings().first() is not None

        except Exception as e:
            self.logger.error(f"Failed to delete todo {todo_id}: {e}")
            return False

    async def get_todos_by_status(self, user_id: str, status: str) -> List[TodoItem]:
        """Get todos filtered by status for a specific user."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(f"""
                        SELECT {_SELECT_TODO_COLS}
                        FROM todos
                        WHERE user_id = :user_id AND status = :status
                        ORDER BY {_ORDER_BY_PRIORITY}
                    """),
                    {"user_id": user_id, "status": status},
                )
                return [self._build_todo(dict(row)) for row in result.mappings()]

        except Exception as e:
            self.logger.error(
                f"Failed to get todos with status {status} for user {user_id}: {e}"
            )
            return []

    async def get_todos_by_conversation(
        self, user_id: str, conversation_id: int
    ) -> List[TodoItem]:
        """Get todos for a specific conversation and user."""
        try:
            async with self.session_factory() as session:
                result = await session.execute(
                    text(f"""
                        SELECT {_SELECT_TODO_COLS}
                        FROM todos
                        WHERE user_id = :user_id AND conversation_id = :conversation_id
                        ORDER BY created_at DESC
                    """),
                    {"user_id": user_id, "conversation_id": conversation_id},
                )
                return [self._build_todo(dict(row)) for row in result.mappings()]

        except Exception as e:
            self.logger.error(
                f"Failed to get todos for conversation {conversation_id} and user {user_id}: {e}"
            )
            return []