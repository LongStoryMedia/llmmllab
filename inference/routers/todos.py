"""
Todo router for handling user todo list management.

"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Request

from db import storage
from middleware.auth import get_user_id
from models.todo_item import TodoItem

router = APIRouter(prefix="/todos", tags=["todos"])


@router.post("/", response_model=TodoItem)
async def create_todo(todo: TodoItem, request: Request):
    """Create a new todo item for the authenticated user"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        return await storage.get_service(storage.todo).add_todo(todo)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating todo: {str(e)}"
        ) from e


@router.get("/", response_model=List[TodoItem])
async def get_todos(request: Request, status: Optional[str] = None):
    """Get all todos for the authenticated user, optionally filtered by status"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        if status:
            todos = await storage.get_service(storage.todo).get_todos_by_status(
                user_id, status
            )
        else:
            todos = await storage.get_service(storage.todo).get_todos_by_user(user_id)

        return todos
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving todos: {str(e)}"
        ) from e


@router.get("/{todo_id}", response_model=TodoItem)
async def get_todo(request: Request, todo_id: int):
    """Get a specific todo item by ID"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        todo = await storage.get_service(storage.todo).get_todo_by_id(todo_id, user_id)

        if not todo:
            raise HTTPException(status_code=404, detail="Todo item not found")

        return todo

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving todo: {str(e)}"
        ) from e


@router.put("/{todo_id}", response_model=TodoItem)
async def update_todo(todo: TodoItem, request: Request):
    """Update a specific todo item"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        return await storage.get_service(storage.todo).update_todo(todo)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error updating todo: {str(e)}"
        ) from e


@router.delete("/{todo_id}")
async def delete_todo(request: Request, todo_id: int):
    """Delete a specific todo item"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        success = await storage.get_service(storage.todo).delete_todo(todo_id, user_id)

        if not success:
            raise HTTPException(status_code=404, detail="Todo item not found")

        return {"message": "Todo item deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting todo: {str(e)}"
        ) from e


@router.get("/conversation/{conversation_id}", response_model=List[TodoItem])
async def get_todos_by_conversation(request: Request, conversation_id: int):
    """Get all todos for a specific conversation"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        todos = await storage.get_service(storage.todo).get_todos_by_conversation(
            user_id, conversation_id
        )
        return todos
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving todos: {str(e)}"
        ) from e


@router.get("/status/{status}", response_model=List[TodoItem])
async def get_todos_by_status(request: Request, status: str):
    """Get all todos filtered by status for the authenticated user"""
    user_id = get_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        return await storage.get_service(storage.todo).get_todos_by_status(
            user_id, status
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving todos: {str(e)}"
        ) from e
