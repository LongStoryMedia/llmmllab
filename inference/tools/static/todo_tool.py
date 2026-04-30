"""Todo management tool for LangChain workflows."""

from langchain.tools import tool
from models import TodoItem
from pydantic import BaseModel, Field
from typing import List


class TodoListResponse(BaseModel):
    """Grammar-constrained response for todo tool."""

    todos: List[TodoItem] = Field(..., description="List of todos for state update")
    message: str = Field(..., description="Confirmation message with updated todo list")


@tool
def write_todos(todos: list[dict | TodoItem]) -> TodoListResponse:
    """
    Create and manage a structured task list for your current work session. This tool is grammar-constrained and expects a list of todos as input and returns a TodoListResponse as output.

    Usage instructions for LLMs:
    - Always call this tool FIRST when managing 3 or more tasks.
    - Input must be a list of todos, each with: title, status, priority, and optionally description, due_date.
    - Status must be one of: 'not-started', 'in-progress', 'completed', 'cancelled'.
    - Priority must be one of: 'low', 'medium', 'high', 'urgent'.
    - Output is a TodoListResponse containing todos (for state update) and a confirmation message.

    Args:
        todos: List of dicts or TodoItem objects representing tasks.

    Returns:
        TodoListResponse: Structured todos and confirmation message for state update.
    """
    if not todos:
        return TodoListResponse(todos=[], message="No todos provided")

    # Coerce input to TodoItem and validate
    valid_todos = []
    for todo in todos:
        if isinstance(todo, dict):
            try:
                item = TodoItem(**todo)
            except Exception:
                continue
        elif isinstance(todo, TodoItem):
            item = todo
        else:
            continue
        valid_todos.append(item)

    formatted_todos = [f"- [{td.status}] {td.title}" for td in valid_todos]
    todo_text = "\n".join(formatted_todos)
    message = (
        f"Updated todo list to:\n{todo_text}"
        if valid_todos
        else "No valid todos provided"
    )
    return TodoListResponse(todos=valid_todos, message=message)
