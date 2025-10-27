"""
Message storage service with enhanced support for tool_calls and thoughts.
Handles message persistence, caching, and proper aggregation of related data.
"""

import asyncpg
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.tool_call import ToolCall
from models.thought import Thought
from models.resource_usage import ResourceUsage
from models.intent_analysis import IntentAnalysis
from db.cache_storage import cache_storage
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_storage")


class MessageStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="message_storage_instance")

    async def add_message(self, message: Message) -> Optional[int]:
        """
        Add a message with all its related content, tool_calls, and thoughts.
        Uses proper transaction handling for data consistency.
        """
        if not message.conversation_id:
            raise ValueError("Message must have a conversation_id")

        async with self.typed_pool.acquire() as conn:
            async with conn.transaction():
                # Insert the main message record
                row = await conn.fetchrow(
                    self.get_query("message.add_message"),
                    message.conversation_id,
                    message.role,
                )
                message_id = row["id"] if row and "id" in row else None

                if not message_id:
                    self.logger.error("Failed to get message_id after insert")
                    return None

                # Insert message contents
                if message.content:
                    await self._insert_message_contents(
                        conn, message_id, message.content
                    )

                # Insert tool_calls if present
                if message.tool_calls:
                    await self._insert_tool_calls(conn, message_id, message.tool_calls)

                # Insert thoughts if present
                if message.thoughts:
                    await self._insert_thoughts(conn, message_id, message.thoughts)

                # Set the message_id on the message object
                message.id = message_id

                # Cache and invalidate appropriately
                try:
                    cache_storage.cache_message(message)
                    cache_storage.invalidate_conversation_messages_cache(
                        message.conversation_id
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to update cache for message {message_id}: {e}"
                    )

                return message_id

    async def get_message(self, message_id: int) -> Optional[Message]:
        """
        Get a message by ID with all related content, tool_calls, and thoughts.
        """
        # First try to get from cache
        cached_message = cache_storage.get_message_from_cache(message_id)
        if cached_message:
            return cached_message

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(self.get_query("message.get_message"), message_id)
            if not row:
                return None

            # Parse all related data from the row
            message_data = self._parse_message_row(dict(row))

            message = Message(**message_data)

            # Cache the result for future use
            try:
                cache_storage.cache_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to cache message {message_id}: {e}")

            return message

    async def get_conversation_history(self, conversation_id: int) -> List[Message]:
        """
        Gets messages for a conversation, ordered and without messages that have been summarized already.
        """
        # First try to get from cache
        cached_messages = cache_storage.get_conversation_messages(conversation_id)
        if cached_messages:
            return self._validate_cached_messages(cached_messages)

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("message.get_conversation_history"),
                conversation_id,
            )
            message_dicts = [dict(row) for row in rows]

            messages = (
                await self._build_messages(conversation_id, message_dicts, conn)
                if message_dicts
                else []
            )

            if messages:
                try:
                    cache_storage.cache_conversation_messages(conversation_id, messages)
                except Exception as e:
                    self.logger.warning(f"Failed to cache conversation messages: {e}")

            return messages

    async def get_messages_by_conversation_id(
        self, conversation_id: int, limit: int, offset: int
    ) -> List[Message]:
        """
        Gets messages for a conversation by conversation_id with pagination.
        """
        # Check cache first
        cached_messages = cache_storage.get_messages_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_messages:
            return cached_messages

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("message.get_by_conversation_id"),
                conversation_id,
                limit,
                offset,
            )
            message_dicts = [dict(row) for row in rows]
            messages = (
                await self._build_messages(conversation_id, message_dicts, conn)
                if message_dicts
                else []
            )

            if messages:
                try:
                    cache_storage.cache_messages_by_conversation_id(
                        conversation_id, messages
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to cache paginated messages: {e}")

            return messages

    async def delete_message(self, message_id: int) -> None:
        """
        Delete a message and all its related data.
        Cascade delete triggers handle related table cleanup automatically.
        """
        # Get the message to find its conversation_id before deletion
        message = await self.get_message(message_id)
        if not message:
            self.logger.warning(
                f"Message {message_id} not found and could not be deleted"
            )
            return

        async with self.typed_pool.acquire() as conn:
            async with conn.transaction():
                # Delete the message - cascade triggers will handle related data
                # (message_contents, tool_calls, thoughts, analyses, etc.)
                await conn.execute(
                    self.get_query("message.delete_message_record"), message_id
                )
                self.logger.info(
                    f"Deleted message {message_id} and related data from database"
                )

        # Invalidate caches
        try:
            cache_storage.invalidate_message_cache(message_id)
            if message.conversation_id:
                cache_storage.invalidate_conversation_messages_cache(
                    message.conversation_id
                )
        except Exception as e:
            self.logger.warning(
                f"Failed to invalidate cache for deleted message {message_id}: {e}"
            )

    async def bulk_delete_messages_from_timestamp(
        self, conversation_id: int, from_timestamp: datetime
    ) -> int:
        """
        Delete all messages in a conversation created at or after the specified timestamp.
        This is more efficient than deleting messages one by one, especially with TimescaleDB.
        Cascade delete triggers automatically handle related data (message_contents, thoughts, tool_calls, analyses).

        Args:
            conversation_id: The conversation ID
            from_timestamp: Delete messages created at or after this timestamp

        Returns:
            Number of messages deleted
        """
        async with self.typed_pool.acquire() as conn:
            async with conn.transaction():
                # Delete messages - cascade triggers will automatically delete related data
                # (message_contents, thoughts, tool_calls, analyses, etc.)
                message_result = await conn.execute(
                    self.get_query("message.delete_messages_from_timestamp"),
                    conversation_id,
                    from_timestamp,
                )

                # Extract the number of deleted rows from the command result
                deleted_count = (
                    int(message_result.split()[-1])
                    if message_result and message_result.split()
                    else 0
                )

                logger.info(
                    f"Bulk deleted {deleted_count} messages from conversation {conversation_id} created >= {from_timestamp} (cascade triggers handled related data)"
                )

        # Invalidate conversation messages list cache
        cache_storage.invalidate_conversation_messages_cache(conversation_id)

        return deleted_count

    async def _build_messages(
        self, conversation_id: int, message_dicts: List[dict], conn: TypedConnection
    ) -> List[Message]:
        """Build Message objects from database rows with all related data."""
        messages: List[Message] = []
        for msg_dict in message_dicts:
            try:
                # Ensure conversation_id is set
                if (
                    "conversation_id" not in msg_dict
                    or msg_dict["conversation_id"] is None
                ):
                    msg_dict["conversation_id"] = conversation_id

                # Parse all message data using the unified parser
                message_data = self._parse_message_row(msg_dict)

                # Create the Message object
                message_obj = Message(**message_data)
                messages.append(message_obj)
            except Exception as e:
                self.logger.warning(
                    f"Failed to create Message object: {e}, msg={msg_dict}"
                )

        return messages

    async def _insert_message_contents(
        self, conn: TypedConnection, message_id: int, contents: List[MessageContent]
    ) -> None:
        """Helper method to insert message contents."""
        for content in contents:
            await conn.execute(
                self.get_query("message_content.add_content"),
                message_id,
                content.type,
                content.text,
                content.url,
            )

    async def _insert_tool_calls(
        self, conn: TypedConnection, message_id: int, tool_calls: List[ToolCall]
    ) -> None:
        """Helper method to insert tool calls."""
        for tool_call in tool_calls:
            # Convert resource_usage to dict if it's a ResourceUsage object
            resource_usage_dict = None
            if tool_call.resource_usage:
                if hasattr(tool_call.resource_usage, "dict"):
                    resource_usage_dict = tool_call.resource_usage.dict()
                else:
                    resource_usage_dict = tool_call.resource_usage

            await conn.execute(
                self.get_query("tool_call.add_tool_call"),
                message_id,
                tool_call.tool_name,
                tool_call.execution_id,
                tool_call.success,
                json.dumps(tool_call.args) if tool_call.args else None,
                json.dumps(tool_call.result_data) if tool_call.result_data else None,
                tool_call.error_message,
                tool_call.execution_time_ms,
                json.dumps(resource_usage_dict) if resource_usage_dict else None,
            )

    async def _insert_thoughts(
        self, conn: TypedConnection, message_id: int, thoughts: List[Thought]
    ) -> None:
        """Helper method to insert thoughts."""
        for thought in thoughts:
            await conn.execute(
                self.get_query("thought.add_thought"),
                message_id,
                thought.text,
            )

    def _parse_message_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a database row containing message data with aggregated JSON fields.
        Handles contents, tool_calls, thoughts, and analyses aggregation.
        """
        # Parse message contents from JSON array
        contents = self._parse_contents(row.get("contents"))

        # Parse tool_calls from JSON array
        tool_calls = self._parse_tool_calls(row.get("tool_calls"))

        # Parse thoughts from JSON array
        thoughts = self._parse_thoughts(row.get("thoughts"))

        # Parse analyses from JSON array
        analyses = self._parse_analyses(row.get("analyses"))

        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "role": row["role"],
            "created_at": row["created_at"],
            "content": contents,
            "tool_calls": tool_calls if tool_calls else None,
            "thoughts": thoughts if thoughts else None,
            "analyses": analyses if analyses else None,
        }

    def _parse_contents(self, contents_data: Any) -> List[MessageContent]:
        """Parse message contents from JSON data."""
        if not contents_data:
            return [MessageContent(type=MessageContentType.TEXT, text="", url=None)]

        contents = []
        # Parse JSON data (could be string or already parsed)
        if isinstance(contents_data, str):
            parsed_data = json.loads(contents_data)
        else:
            parsed_data = contents_data

        for content_data in parsed_data:
            contents.append(
                MessageContent(
                    type=MessageContentType(
                        content_data.get("type", MessageContentType.TEXT)
                    ),
                    text=content_data.get("text_content", ""),
                    url=content_data.get("url"),
                )
            )

        return contents

    def _parse_tool_calls(self, tool_calls_data: Any) -> Optional[List[ToolCall]]:
        """Parse tool_calls from JSON data."""
        if not tool_calls_data:
            return None

        tool_calls = []
        # Parse JSON data (could be string or already parsed)
        if isinstance(tool_calls_data, str):
            parsed_data = json.loads(tool_calls_data)
        else:
            parsed_data = tool_calls_data

        for tc_data in parsed_data:
            # Parse resource_usage if present
            resource_usage = None
            if tc_data.get("resource_usage"):
                try:
                    resource_usage = ResourceUsage(**tc_data["resource_usage"])
                except Exception as e:
                    self.logger.warning(f"Failed to parse resource_usage: {e}")

            tool_calls.append(
                ToolCall(
                    tool_name=tc_data["tool_name"],
                    execution_id=tc_data.get("execution_id"),
                    success=tc_data["success"],
                    args=tc_data.get("args"),
                    result_data=tc_data.get("result_data"),
                    error_message=tc_data.get("error_message"),
                    execution_time_ms=tc_data.get("execution_time_ms"),
                    resource_usage=resource_usage,
                )
            )

        return tool_calls if tool_calls else None

    def _parse_thoughts(self, thoughts_data: Any) -> Optional[List[Thought]]:
        """Parse thoughts from JSON data."""
        if not thoughts_data:
            return None

        thoughts = []
        # Parse JSON data (could be string or already parsed)
        if isinstance(thoughts_data, str):
            parsed_data = json.loads(thoughts_data)
        else:
            parsed_data = thoughts_data

        for th_data in parsed_data:
            thoughts.append(
                Thought(
                    id=th_data.get("id"),
                    message_id=th_data.get("message_id"),
                    text=th_data["text"],
                    created_at=th_data.get("created_at"),
                )
            )

        return thoughts if thoughts else None

    def _parse_analyses(self, analyses_data: Any) -> Optional[List[IntentAnalysis]]:
        """Parse analyses from JSON data."""
        if not analyses_data:
            return None

        analyses = []
        # Parse JSON data (could be string or already parsed)
        if isinstance(analyses_data, str):
            parsed_data = json.loads(analyses_data)
        else:
            parsed_data = analyses_data

        for analysis_data in parsed_data:
            try:
                # Convert JSON fields back to proper types
                from models.workflow_type import WorkflowType
                from models.complexity_level import ComplexityLevel
                from models.required_capability import RequiredCapability
                from models.computational_requirement import ComputationalRequirement

                # Parse enums and JSON fields
                workflow_type = WorkflowType(analysis_data.get("workflow_type", "UNKNOWN"))
                complexity_level = ComplexityLevel(analysis_data.get("complexity_level", "LOW"))
                
                required_capabilities = []
                if analysis_data.get("required_capabilities"):
                    for cap in analysis_data["required_capabilities"]:
                        try:
                            required_capabilities.append(RequiredCapability(cap))
                        except (ValueError, TypeError):
                            pass  # Skip invalid capabilities
                
                computational_requirements = ComputationalRequirement(
                    analysis_data.get("computational_requirements", "MINIMAL")
                )

                analysis = IntentAnalysis(
                    workflow_type=workflow_type,
                    complexity_level=complexity_level,
                    required_capabilities=required_capabilities,
                    domain_specificity=float(analysis_data.get("domain_specificity", 0.0)),
                    reusability_potential=float(analysis_data.get("reusability_potential", 0.0)),
                    confidence=float(analysis_data.get("confidence", 0.0)),
                    response_format=analysis_data.get("response_format"),
                    technical_domain=analysis_data.get("technical_domain"),
                    requires_tools=bool(analysis_data.get("requires_tools", False)),
                    requires_custom_tools=bool(analysis_data.get("requires_custom_tools", False)),
                    tool_complexity_score=float(analysis_data.get("tool_complexity_score", 0.0)),
                    computational_requirements=computational_requirements,
                )
                analyses.append(analysis)
            except Exception as e:
                self.logger.warning(f"Failed to parse analysis data: {e}")
                continue

        return analyses if analyses else None

    def _validate_cached_messages(self, cached_messages: Any) -> List[Message]:
        """Validate and clean cached messages data."""
        validated_messages = []

        # Ensure cached_messages is iterable
        if not isinstance(cached_messages, list):
            cached_messages = [cached_messages]

        for msg in cached_messages:
            # Ensure content is a list - this handles legacy cached data
            if not msg.content:
                msg.content = [MessageContent(type=MessageContentType.TEXT, text="")]
            elif not isinstance(msg.content, list):
                msg.content = [
                    MessageContent(type=MessageContentType.TEXT, text=str(msg.content))
                ]

            validated_messages.append(msg)

        return validated_messages
