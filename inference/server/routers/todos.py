"""
Todo router for handling user todo list management.

Note: This router is included in app.py with both non-versioned and versioned paths:
- Non-versioned: /todos/...
- Versioned: /v1/todos/...
"""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from db import storage
from server.middleware.auth import get_user_id
from models.todo_item import TodoItem

router = APIRouter(prefix="/todos", tags=["todos"])


class CreateTodoRequest(BaseModel):
    """Request model for creating a new todo item"""
    title: str
    description: Optional[str] = None
    status: str = "not-started"
    priority: str = "medium"
    due_date: Optional[datetime] = None


class UpdateTodoRequest(BaseModel):
    """Request model for updating a todo item"""
    title: str
    description: Optional[str] = None
    status: str
    priority: str
    due_date: Optional[datetime] = None


@router.post("/", response_model=TodoItem)
async def create_todo(request: Request, todo_request: CreateTodoRequest):
    """Create a new todo item for the authenticated user"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.todo:
        raise HTTPException(status_code=503, detail="Todo storage not initialized")

    # Validate status and priority
    valid_statuses = ["not-started", "in-progress", "completed", "cancelled"]
    valid_priorities = ["low", "medium", "high", "urgent"]

    if todo_request.status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    if todo_request.priority not in valid_priorities:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid priority. Must be one of: {', '.join(valid_priorities)}"
        )

    try:
        todo = await storage.todo.add_todo(
            user_id=user_id,
            title=todo_request.title,
            description=todo_request.description,
            status=todo_request.status,
            priority=todo_request.priority,
            due_date=todo_request.due_date,
        )

        if not todo:
            raise HTTPException(status_code=500, detail="Failed to create todo item")

        return todo

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating todo: {str(e)}")


@router.get("/", response_model=List[TodoItem])
async def get_todos(request: Request, status: Optional[str] = None):
    """Get all todos for the authenticated user, optionally filtered by status"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.todo:
        raise HTTPException(status_code=503, detail="Todo storage not initialized")

    # Validate status filter if provided
    if status:
        valid_statuses = ["not-started", "in-progress", "completed", "cancelled"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid status filter. Must be one of: {', '.join(valid_statuses)}"
            )

    try:
        if status:
            todos = await storage.todo.get_todos_by_status(user_id, status)
        else:
            todos = await storage.todo.get_todos_by_user(user_id)

        return todos

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving todos: {str(e)}")


@router.get("/{todo_id}", response_model=TodoItem)
async def get_todo(request: Request, todo_id: int):
    """Get a specific todo item by ID"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.todo:
        raise HTTPException(status_code=503, detail="Todo storage not initialized")

    try:
        todo = await storage.todo.get_todo_by_id(todo_id, user_id)

        if not todo:
            raise HTTPException(status_code=404, detail="Todo item not found")

        return todo

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving todo: {str(e)}")


@router.put("/{todo_id}", response_model=TodoItem)
async def update_todo(request: Request, todo_id: int, todo_request: UpdateTodoRequest):
    """Update a specific todo item"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.todo:
        raise HTTPException(status_code=503, detail="Todo storage not initialized")

    # Validate status and priority
    valid_statuses = ["not-started", "in-progress", "completed", "cancelled"]
    valid_priorities = ["low", "medium", "high", "urgent"]

    if todo_request.status not in valid_statuses:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
        )

    if todo_request.priority not in valid_priorities:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid priority. Must be one of: {', '.join(valid_priorities)}"
        )

    try:
        todo = await storage.todo.update_todo(
            todo_id=todo_id,
            user_id=user_id,
            title=todo_request.title,
            description=todo_request.description,
            status=todo_request.status,
            priority=todo_request.priority,
            due_date=todo_request.due_date,
        )

        if not todo:
            raise HTTPException(status_code=404, detail="Todo item not found")

        return todo

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating todo: {str(e)}")


@router.delete("/{todo_id}")
async def delete_todo(request: Request, todo_id: int):
    """Delete a specific todo item"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not storage.initialized:
        raise HTTPException(status_code=503, detail="Database not initialized")

    if not storage.todo:
        raise HTTPException(status_code=503, detail="Todo storage not initialized")

    try:
        success = await storage.todo.delete_todo(todo_id, user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Todo item not found")

        return {"message": "Todo item deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting todo: {str(e)}")