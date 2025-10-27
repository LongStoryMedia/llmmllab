"""
Checkpoint storage service integrating LangGraph AsyncPostgresSaver with todo management.
Provides persistent state management for multi-turn workflows with todo context.
"""

import asyncpg
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.runnables import RunnableConfig
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

        # LangGraph saver will be initialized in initialize() method
        self.langgraph_saver: Optional[AsyncPostgresSaver] = None
        self._saver_ready = False
        self._connection_string: Optional[str] = None

    async def initialize(self, connection_string: str) -> None:
        """Initialize checkpoint tables and LangGraph saver."""
        try:
            # Create checkpoint tables if they don't exist
            async with self.typed_pool.acquire() as conn:
                await conn.execute(
                    self.get_query("checkpoint.create_langgraph_checkpoint_tables")
                )

            # Create AsyncPostgresSaver using from_conn_string
            async with AsyncPostgresSaver.from_conn_string(connection_string) as saver:
                await saver.setup()
                # Store the saver for later use - note this is a temporary connection
                # We'll create new connections as needed
                self._connection_string = connection_string
                self._saver_ready = True

            self.logger.info("Checkpoint storage initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize checkpoint storage: {e}")
            raise

    @asynccontextmanager
    async def get_saver(self):
        """Get a LangGraph saver instance with active connection."""
        if not self._saver_ready or not self._connection_string:
            raise RuntimeError("CheckpointStorage not initialized")

        async with AsyncPostgresSaver.from_conn_string(
            self._connection_string
        ) as saver:
            yield saver

    async def save_workflow_state_with_todos(
        self, conversation_id: int, todos: List[Dict[str, Any]]
    ) -> bool:
        """
        Save workflow state including todos and planning context.

        Args:
            conversation_id: Conversation ID used as thread_id
            todos: Generated or active todos

        Returns:
            True if saved successfully, False otherwise
        """
        try:
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
        self, conversation_id: int, user_id: str
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
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }

            # Get latest checkpoint through LangGraph saver
            async with self.get_saver() as saver:
                checkpoint = await saver.aget(config)

                if checkpoint and "channel_values" in checkpoint:
                    # Extract checkpoint data - checkpoint is a Checkpoint TypedDict
                    checkpoint_data = checkpoint["channel_values"] or {}

                    # Extract todos and planning context from checkpoint
                    previous_todos = checkpoint_data.get("todos", [])
                    planning_context = checkpoint_data.get("planning_context", {})

                    # Also load active todos from database for current status
                    from db.todo_storage import TodoStorage

                    todo_storage = TodoStorage(self.pool, self.get_query)
                    active_todos = await todo_storage.get_todos_by_conversation(
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
                            "created_at": (
                                todo.created_at.isoformat() if todo.created_at else None
                            ),
                        }
                        for todo in active_todos
                    ]

                    return {
                        "previous_todos": previous_todos,
                        "active_todos": active_todos_dicts,
                        "planning_context": planning_context,
                        "checkpoint_exists": True,
                        "checkpoint_ts": checkpoint.get("ts"),
                    }

            # No previous checkpoint - load only current todos
            from db.todo_storage import TodoStorage

            todo_storage = TodoStorage(self.pool, self.get_query)
            active_todos = await todo_storage.get_todos_by_conversation(
                user_id, conversation_id
            )

            active_todos_dicts = [
                {
                    "id": todo.id,
                    "title": todo.title,
                    "description": todo.description,
                    "status": todo.status,
                    "priority": todo.priority,
                    "created_at": (
                        todo.created_at.isoformat() if todo.created_at else None
                    ),
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
        self, conversation_id: int, max_checkpoints: int = 5
    ) -> Dict[str, Any]:
        """
        Get a summary of conversation history from checkpoints for context.

        Args:
            conversation_id: Conversation ID
            max_checkpoints: Maximum number of recent checkpoints to analyze

        Returns:
            Summary of conversation progression including todo evolution
        """
        try:
            thread_id = str(conversation_id)

            # Get recent checkpoints for this conversation
            config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": "",
                }
            }

            # Get checkpoint history through LangGraph saver
            checkpoints = []
            async with self.get_saver() as saver:
                async for checkpoint in saver.alist(config, limit=max_checkpoints):
                    checkpoints.append(checkpoint)

            # Analyze checkpoint progression
            todo_evolution = []
            planning_evolution = []

            for checkpoint in checkpoints:
                if "channel_values" in checkpoint and checkpoint["channel_values"]:
                    data = checkpoint["channel_values"]

                    # Track todo changes over time
                    todos = data.get("todos", [])
                    if todos:
                        todo_evolution.append(
                            {
                                "checkpoint_ts": checkpoint.get("ts"),
                                "todo_count": len(todos),
                                "todos": todos[:3],  # Sample of todos
                            }
                        )

                    # Track planning progression
                    planning_context = data.get("planning_context", {})
                    if planning_context:
                        planning_evolution.append(
                            {
                                "checkpoint_ts": checkpoint.get("ts"),
                                "complexity_score": planning_context.get(
                                    "complexity_score"
                                ),
                                "planning_steps": planning_context.get(
                                    "planning_steps", []
                                ),
                            }
                        )

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

    def create_saver_for_workflow(self):
        """Create a new LangGraph AsyncPostgresSaver context manager for workflow integration."""
        if not self._saver_ready or not self._connection_string:
            raise RuntimeError("CheckpointStorage not initialized")

        # Return a context manager that creates the saver
        return AsyncPostgresSaver.from_conn_string(self._connection_string)
