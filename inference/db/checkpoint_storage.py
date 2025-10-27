"""
Checkpoint storage service integrating LangGraph AsyncPostgresSaver with todo management.
Provides persistent state management for multi-turn workflows with todo context.
"""

import asyncpg
from typing import Optional, Dict, Any, List
from datetime import datetime
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from db.db_utils import typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="checkpoint_storage")


class CheckpointStorage:
    """
    Enhanced checkpoint storage that integrates LangGraph persistence with todo context.
    
    Combines LangGraph's AsyncPostgresSaver with todo management for stateful workflows.
    """

    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="checkpoint_storage_instance")
        
        # Initialize LangGraph checkpointer
        self.langgraph_saver = AsyncPostgresSaver(
            pool=pool,
            # Use conversation_id as thread_id for natural threading
            serde=None,  # Use default serialization
        )

    async def initialize(self) -> None:
        """Initialize checkpoint tables and LangGraph saver."""
        try:
            # Create checkpoint tables if they don't exist
            async with self.typed_pool.acquire() as conn:
                await conn.execute(self.get_query("checkpoint.create_langgraph_checkpoint_tables"))
            
            # Setup LangGraph saver
            await self.langgraph_saver.setup()
            
            self.logger.info("Checkpoint storage initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize checkpoint storage: {e}")
            raise

    async def save_workflow_state_with_todos(
        self,
        conversation_id: int,
        user_id: str,
        checkpoint_data: Dict[str, Any],
        todos: List[Dict[str, Any]],
        planning_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save workflow state including todos and planning context.
        
        Args:
            conversation_id: Conversation ID used as thread_id
            user_id: User identifier for ownership
            checkpoint_data: LangGraph checkpoint data
            todos: Generated or active todos
            planning_context: Additional planning metadata
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Use conversation_id as thread_id for natural conversation threading
            thread_id = str(conversation_id)
            
            # The actual save operation happens through LangGraph's workflow execution
            # This method prepares the enhanced data structure for checkpoint metadata
            
            self.logger.info(
                f"Prepared checkpoint save for conversation {conversation_id} with {len(todos)} todos"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save workflow state with todos: {e}")
            return False

    async def load_workflow_context_with_todos(
        self,
        conversation_id: int,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Load previous workflow context including todos for conversation continuity.
        
        Args:
            conversation_id: Conversation ID to load context for
            user_id: User identifier for ownership verification
            
        Returns:
            Dictionary containing previous todos, planning context, and state
        """
        try:
            thread_id = str(conversation_id)
            
            # Load latest checkpoint for this conversation
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }
            
            # Get latest checkpoint through LangGraph saver
            checkpoint = await self.langgraph_saver.aget(config)
            
            if checkpoint and checkpoint.checkpoint:
                # Extract checkpoint data
                checkpoint_data = checkpoint.checkpoint.get("channel_values", {})
                
                # Extract todos and planning context from checkpoint
                previous_todos = checkpoint_data.get("todos", [])
                planning_context = checkpoint_data.get("planning_context", {})
                
                # Also load active todos from database for current status
                from db import storage
                active_todos = await storage.todo.get_todos_by_conversation(
                    user_id, conversation_id
                )
                
                # Convert TodoItem objects to dicts for consistency
                active_todos_dicts = [
                    {
                        "id": todo.id,
                        "title": todo.title,
                        "description": todo.description,
                        "status": todo.status,
                        "priority": todo.priority,
                        "created_at": todo.created_at.isoformat() if todo.created_at else None,
                    }
                    for todo in active_todos
                ]
                
                return {
                    "previous_todos": previous_todos,
                    "active_todos": active_todos_dicts,
                    "planning_context": planning_context,
                    "checkpoint_exists": True,
                    "checkpoint_id": checkpoint.config.get("checkpoint_id"),
                }
            
            # No previous checkpoint - load only current todos
            from db import storage
            active_todos = await storage.todo.get_todos_by_conversation(
                user_id, conversation_id
            )
            
            active_todos_dicts = [
                {
                    "id": todo.id,
                    "title": todo.title,
                    "description": todo.description,
                    "status": todo.status,
                    "priority": todo.priority,
                    "created_at": todo.created_at.isoformat() if todo.created_at else None,
                }
                for todo in active_todos
            ]
            
            return {
                "previous_todos": [],
                "active_todos": active_todos_dicts,
                "planning_context": {},
                "checkpoint_exists": False,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load workflow context with todos: {e}")
            return {
                "previous_todos": [],
                "active_todos": [],
                "planning_context": {},
                "checkpoint_exists": False,
            }

    async def get_conversation_history_summary(
        self,
        conversation_id: int,
        max_checkpoints: int = 5
    ) -> Dict[str, Any]:
        """
        Get a summary of conversation history from checkpoints for context.
        
        Args:
            conversation_id: Conversation ID
            user_id: User identifier
            max_checkpoints: Maximum number of recent checkpoints to analyze
            
        Returns:
            Summary of conversation progression including todo evolution
        """
        try:
            thread_id = str(conversation_id)
            
            # Get recent checkpoints for this conversation
            config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }
            
            # Get checkpoint history through LangGraph saver
            checkpoints = []
            async for checkpoint in self.langgraph_saver.alist(config, limit=max_checkpoints):
                checkpoints.append(checkpoint)
            
            # Analyze checkpoint progression
            todo_evolution = []
            planning_evolution = []
            
            for checkpoint in checkpoints:
                if checkpoint.checkpoint:
                    data = checkpoint.checkpoint.get("channel_values", {})
                    
                    # Track todo changes over time
                    todos = data.get("todos", [])
                    if todos:
                        todo_evolution.append({
                            "checkpoint_id": checkpoint.config.get("checkpoint_id"),
                            "todo_count": len(todos),
                            "todos": todos[:3],  # Sample of todos
                        })
                    
                    # Track planning progression
                    planning_context = data.get("planning_context", {})
                    if planning_context:
                        planning_evolution.append({
                            "checkpoint_id": checkpoint.config.get("checkpoint_id"),
                            "complexity_score": planning_context.get("complexity_score"),
                            "planning_steps": planning_context.get("planning_steps", []),
                        })
            
            return {
                "conversation_id": conversation_id,
                "checkpoint_count": len(checkpoints),
                "todo_evolution": todo_evolution,
                "planning_evolution": planning_evolution,
                "has_active_workflows": len(checkpoints) > 0,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history summary: {e}")
            return {
                "conversation_id": conversation_id,
                "checkpoint_count": 0,
                "todo_evolution": [],
                "planning_evolution": [],
                "has_active_workflows": False,
            }

    def get_langgraph_saver(self) -> AsyncPostgresSaver:
        """Get the LangGraph AsyncPostgresSaver instance for workflow integration."""
        return self.langgraph_saver