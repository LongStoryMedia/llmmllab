"""
Storage service for managing tool call entities in the database.
Tool calls represent execution results from tools associated with messages.
"""

import asyncpg
from typing import List, Optional
from datetime import datetime
from models.tool_execution_result import ToolExecutionResult
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="tool_call_storage")


class ToolCallStorage:
    """Storage service for tool call entities with CRUD operations."""
    
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="tool_call_storage_instance")

    async def add_tool_call(
        self,
        message_id: int,
        tool_data: dict,
        created_at: Optional[datetime] = None,
    ) -> Optional[int]:
        """
        Add a new tool call to the database.
        
        Args:
            message_id: ID of the associated message
            tool_data: The tool execution data as JSON
            created_at: Optional timestamp (defaults to NOW())
            
        Returns:
            The ID of the created tool call, or None on failure
        """
        if created_at is None:
            created_at = datetime.utcnow()
            
        try:
            async with self.typed_pool.acquire() as conn:
                row = await conn.fetchrow(
                    self.get_query("tool_call.add_tool_call"),
                    message_id,
                    tool_data,
                    created_at
                )
                
                if row:
                    tool_call_id = row["id"]
                    self.logger.info(f"Added tool call {tool_call_id} for message {message_id}")
                    return tool_call_id
                else:
                    self.logger.error(f"Failed to add tool call for message {message_id}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error adding tool call for message {message_id}: {e}")
            return None

    async def get_tool_calls_by_message(self, message_id: int) -> List[dict]:
        """
        Retrieve all tool calls associated with a message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            List of tool call dictionaries
        """
        try:
            async with self.typed_pool.acquire() as conn:
                rows = await conn.fetch(
                    self.get_query("tool_call.get_by_message"),
                    message_id
                )
                
                tool_calls = []
                for row in rows:
                    tool_call = {
                        "id": row["id"],
                        "message_id": row["message_id"],
                        "tool_data": row["tool_data"],
                        "created_at": row["created_at"]
                    }
                    tool_calls.append(tool_call)
                    
                self.logger.debug(f"Retrieved {len(tool_calls)} tool calls for message {message_id}")
                return tool_calls
                
        except Exception as e:
            self.logger.error(f"Error retrieving tool calls for message {message_id}: {e}")
            return []

    async def delete_tool_calls_by_message(self, message_id: int) -> bool:
        """
        Delete all tool calls associated with a message.
        
        Args:
            message_id: ID of the message
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self.typed_pool.acquire() as conn:
                result = await conn.execute(
                    self.get_query("tool_call.delete_by_message"),
                    message_id
                )
                
                self.logger.info(f"Deleted tool calls for message {message_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting tool calls for message {message_id}: {e}")
            return False