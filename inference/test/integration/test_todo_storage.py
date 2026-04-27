"""Integration tests for TodoStorage — exercises real PostgreSQL."""

from datetime import datetime
from typing import Literal

import pytest
from db.todo_storage import TodoStorage
from models.todo_item import TodoItem


def _make_todo(
    title: str = "Test Todo",
    user_id: str = "user-1",
    status: Literal["not-started", "in-progress", "completed", "cancelled"] = "not-started",
    priority: Literal["low", "medium", "high", "urgent"] = "medium",
    conversation_id: int | None = None,
    description: str | None = None,
    due_date: datetime | None = None,
) -> TodoItem:
    return TodoItem(
        user_id=user_id,
        conversation_id=conversation_id,
        title=title,
        description=description,
        status=status,
        priority=priority,
        due_date=due_date,
    )


@pytest.mark.asyncio
async def test_add_and_get_todo(todo_storage: TodoStorage):
    item = await todo_storage.add_todo(_make_todo("Write tests"))
    assert item is not None
    assert item.id is not None
    assert item.title == "Write tests"
    assert item.status == "not-started"
    assert item.created_at is not None

    fetched = await todo_storage.get_todo_by_id(item.id, "user-1")  # type: ignore[arg-type]
    assert fetched is not None
    assert fetched.title == "Write tests"


@pytest.mark.asyncio
async def test_get_todo_wrong_user(todo_storage: TodoStorage):
    """Getting a todo with a different user_id should return None."""
    item = await todo_storage.add_todo(_make_todo(user_id="user-a"))
    assert item is not None
    result = await todo_storage.get_todo_by_id(item.id, "user-b")  # type: ignore[arg-type]
    assert result is None


@pytest.mark.asyncio
async def test_update_todo(todo_storage: TodoStorage):
    item = await todo_storage.add_todo(_make_todo("Deploy"))
    assert item is not None

    item.status = "completed"
    item.title = "Deployed"
    updated = await todo_storage.update_todo(item)
    assert updated is not None
    assert updated.title == "Deployed"
    assert updated.status == "completed"
    assert updated.updated_at is not None


@pytest.mark.asyncio
async def test_delete_todo(todo_storage: TodoStorage):
    item = await todo_storage.add_todo(_make_todo("Cleanup"))
    assert item is not None

    result = await todo_storage.delete_todo(item.id, "user-1")  # type: ignore[arg-type]
    assert result is True

    fetched = await todo_storage.get_todo_by_id(item.id, "user-1")  # type: ignore[arg-type]
    assert fetched is None


@pytest.mark.asyncio
async def test_delete_todo_wrong_user(todo_storage: TodoStorage):
    item = await todo_storage.add_todo(_make_todo(user_id="owner"))
    assert item is not None
    result = await todo_storage.delete_todo(item.id, "not-owner")  # type: ignore[arg-type]
    assert result is False


@pytest.mark.asyncio
async def test_get_todos_by_user(todo_storage: TodoStorage):
    await todo_storage.add_todo(_make_todo("Task A", priority="low"))
    await todo_storage.add_todo(_make_todo("Task B", priority="urgent"))

    todos = await todo_storage.get_todos_by_user("user-1")
    assert len(todos) == 2
    # Urgent sorts first
    assert todos[0].priority == "urgent"
    assert todos[1].priority == "low"


@pytest.mark.asyncio
async def test_get_todos_by_status(todo_storage: TodoStorage):
    await todo_storage.add_todo(_make_todo("Done", status="completed"))
    await todo_storage.add_todo(_make_todo("Pending", status="not-started"))

    completed = await todo_storage.get_todos_by_status("user-1", "completed")
    assert len(completed) == 1
    assert completed[0].title == "Done"


@pytest.mark.asyncio
async def test_get_todos_by_conversation(todo_storage: TodoStorage):
    await todo_storage.add_todo(_make_todo("Conv 1", conversation_id=10))
    await todo_storage.add_todo(_make_todo("Conv 2", conversation_id=20))
    await todo_storage.add_todo(_make_todo("No conv", conversation_id=None))

    conv1 = await todo_storage.get_todos_by_conversation("user-1", 10)
    assert len(conv1) == 1
    assert conv1[0].title == "Conv 1"


@pytest.mark.asyncio
async def test_todo_with_due_date(todo_storage: TodoStorage):
    due = datetime(2026, 12, 31, 23, 59, 59)
    item = await todo_storage.add_todo(_make_todo("Year end", due_date=due))
    assert item is not None
    assert item.due_date is not None
    assert item.due_date.year == 2026
