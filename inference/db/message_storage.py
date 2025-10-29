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
from models.workflow_type import WorkflowType
from models.complexity_level import ComplexityLevel
from models.required_capability import RequiredCapability
from models.response_format import ResponseFormat
from models.technical_domain import TechnicalDomain
from models.computational_requirement import ComputationalRequirement
from db.cache_storage import cache_storage
from db.db_utils import TypedConnection, typed_pool
from utils.logging import llmmllogger
from db.thought_storage import ThoughtStorage
from db.tool_call_storage import ToolCallStorage
from db.message_content_storage import MessageContentStorage
from db.analysis_storage import AnalysisStorage

logger = llmmllogger.bind(component="message_storage")


class MessageStorage:

    def __init__(
        self,
        pool: asyncpg.Pool,
        get_query,
        thought_storage: ThoughtStorage,
        tool_call_storage: ToolCallStorage,
        message_content_storage: MessageContentStorage,
        analysis_storage: AnalysisStorage,
    ):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = llmmllogger.bind(component="message_storage_instance")

        # Storage service dependencies (will be set after initialization)
        self.thought_storage = thought_storage
        self.tool_call_storage = tool_call_storage
        self.message_content_storage = message_content_storage
        self.analysis_storage = analysis_storage

    async def add_message(
        self,
        message: Message,
        conn: Optional[TypedConnection] = None,
    ) -> Optional[int]:
        """
        Add a message with all its related content, tool_calls, and thoughts.
        Uses proper transaction handling for data consistency.

        Args:
            message: The message to add
            conn: Optional existing connection for transaction support
        """
        if not message.conversation_id:
            raise ValueError("Message must have a conversation_id")

        # Acquire connection if not provided
        if conn is None:
            async with self.typed_pool.acquire() as connection:
                async with connection.transaction():
                    return await self._add_message(message, connection)
        else:
            # Use existing connection (transaction managed externally)
            return await self._add_message(message, conn)

    async def _add_message(
        self,
        message: Message,
        conn: TypedConnection,
    ) -> Optional[int]:
        """
        Internal method to add message using a specific connection.
        """
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
            await self._insert_message_contents(conn, message_id, message.content)

        # Insert tool_calls if present
        if message.tool_calls:
            await self._insert_tool_calls(conn, message_id, message.tool_calls)

        # Insert thoughts if present
        if message.thoughts:
            await self._insert_thoughts(conn, message_id, message.thoughts)

        # Set the message_id on the message object
        message.id = message_id

        # Cache and invalidate appropriately
        # Note: We cache the new message but invalidate conversation cache
        # This ensures cache consistency regardless of transaction state
        if message.conversation_id is not None:
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

    async def get_message(
        self, message_id: int, conn: Optional[TypedConnection] = None
    ) -> Optional[Message]:
        """
        Get a message by ID with all related content, tool_calls, and thoughts using multiple queries.

        Args:
            message_id: The message ID to retrieve
            conn: Optional existing connection for transaction support
        """
        # Try cache first - safe even in transactions for read operations
        cached_message = cache_storage.get_message_from_cache(message_id)
        if cached_message:
            return cached_message

        # Acquire connection if not provided
        if conn is None:
            async with self.typed_pool.acquire() as connection:
                return await self._get_message(
                    message_id, connection, cache_result=True
                )
        else:
            # When using external connection, we still cache but let caller control transaction
            return await self._get_message(message_id, conn, cache_result=True)

    async def _get_message(
        self, message_id: int, conn: TypedConnection, cache_result: bool = True
    ) -> Optional[Message]:
        """
        Internal method to get message using a specific connection.

        Args:
            message_id: The message ID to retrieve
            conn: Database connection to use
            cache_result: Whether to cache the result after fetching
        """
        # Get the base message
        row = await conn.fetchrow(self.get_query("message.get_message"), message_id)
        if not row:
            return None

        message_data = dict(row)

        # Get message contents using separate query
        contents_rows = await conn.fetch(
            self.get_query("message_content.get_by_message"), message_id
        )
        message_data["content"] = [
            MessageContent(**dict(content_row)) for content_row in contents_rows
        ]

        # Get tool calls using separate query
        tool_calls_rows = await conn.fetch(
            self.get_query("tool_call.get_by_message"), message_id
        )
        message_data["tool_calls"] = [
            self._parse_tool_call_row(dict(tool_row)) for tool_row in tool_calls_rows
        ]

        # Get thoughts using separate query
        thoughts_rows = await conn.fetch(
            self.get_query("thought.get_by_message"), message_id
        )
        message_data["thoughts"] = [
            Thought(**dict(thought_row)) for thought_row in thoughts_rows
        ]

        # Get analyses using separate query
        analyses_rows = await conn.fetch(
            self.get_query("analysis.get_by_message"), message_id
        )
        message_data["analyses"] = [
            self._parse_analysis_row(dict(analysis_row))
            for analysis_row in analyses_rows
        ]

        message = Message(**message_data)

        # Cache the result if requested
        if cache_result:
            try:
                cache_storage.cache_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to cache message {message_id}: {e}")

        return message

    def _parse_tool_call_row(self, row: Dict[str, Any]) -> ToolCall:
        """
        Parse an individual tool call row from database into ToolCall object.

        Args:
            row: Database row containing tool call data

        Returns:
            ToolCall object with properly parsed JSON fields
        """
        # Parse JSON fields from strings to dicts
        args = row.get("args", "{}")
        if isinstance(args, str):
            args = json.loads(args)

        result_data = row.get("result_data", "{}")
        if isinstance(result_data, str):
            result_data = json.loads(result_data) if result_data.strip() else {}

        resource_usage_data = row.get("resource_usage", "{}")
        if isinstance(resource_usage_data, str):
            resource_usage_data = (
                json.loads(resource_usage_data) if resource_usage_data.strip() else {}
            )

        # Convert resource_usage dict to ResourceUsage object if not empty
        resource_usage = None
        if resource_usage_data:
            try:
                resource_usage = ResourceUsage(**resource_usage_data)
            except Exception as e:
                self.logger.warning(f"Failed to parse resource_usage: {e}")
                resource_usage = None

        return ToolCall(
            message_id=row.get("message_id"),
            name=row.get("tool_name"),  # Map tool_name from DB to name field in model
            execution_id=row.get("execution_id"),
            success=row.get("success", False),
            args=args,
            result_data=result_data if result_data else None,
            error_message=row.get("error_message"),
            execution_time_ms=row.get("execution_time_ms"),
            resource_usage=resource_usage,
        )

    def _parse_analysis_row(self, row: Dict[str, Any]) -> IntentAnalysis:
        """
        Parse an individual analysis row from database into IntentAnalysis object.

        Args:
            row: Database row containing analysis data

        Returns:
            IntentAnalysis object
        """
        # Parse required_capabilities from JSONB to List[RequiredCapability]
        required_capabilities_data = row.get("required_capabilities", [])
        if isinstance(required_capabilities_data, str):
            required_capabilities_data = json.loads(required_capabilities_data)

        required_capabilities = []
        for cap in required_capabilities_data:
            if isinstance(cap, str):
                try:
                    required_capabilities.append(RequiredCapability(cap))
                except ValueError:
                    self.logger.warning(f"Unknown required capability: {cap}")
            elif hasattr(cap, "value"):  # Enum object
                required_capabilities.append(cap)

        # Parse computational_requirements from JSONB
        comp_req_data = row.get("computational_requirements", {})
        if isinstance(comp_req_data, str):
            comp_req_data = json.loads(comp_req_data)

        computational_requirements = ComputationalRequirement(**comp_req_data)

        # Parse enum fields with fallback
        workflow_type = row.get("workflow_type")
        if isinstance(workflow_type, str):
            try:
                workflow_type = WorkflowType(workflow_type)
            except ValueError:
                workflow_type = WorkflowType.GENERAL  # Fallback

        complexity_level = row.get("complexity_level")
        if isinstance(complexity_level, str):
            try:
                complexity_level = ComplexityLevel(complexity_level)
            except ValueError:
                complexity_level = ComplexityLevel.SIMPLE  # Fallback

        # Parse optional enum fields
        response_format = row.get("response_format")
        if response_format and isinstance(response_format, str):
            try:
                response_format = ResponseFormat(response_format)
            except ValueError:
                response_format = None

        technical_domain = row.get("technical_domain")
        if technical_domain and isinstance(technical_domain, str):
            try:
                technical_domain = TechnicalDomain(technical_domain)
            except ValueError:
                technical_domain = None

        # Ensure required fields are not None
        if workflow_type is None:
            workflow_type = WorkflowType.GENERAL
        if complexity_level is None:
            complexity_level = ComplexityLevel.SIMPLE

        return IntentAnalysis(
            id=row.get("id"),
            message_id=row.get("message_id"),
            workflow_type=workflow_type,
            complexity_level=complexity_level,
            required_capabilities=required_capabilities,
            domain_specificity=row.get("domain_specificity", 0.0),
            reusability_potential=row.get("reusability_potential", 0.0),
            confidence=row.get("confidence", 0.0),
            response_format=response_format,
            technical_domain=technical_domain,
            requires_tools=row.get("requires_tools", False),
            requires_custom_tools=row.get("requires_custom_tools", False),
            tool_complexity_score=row.get("tool_complexity_score", 0.0),
            computational_requirements=computational_requirements,
            created_at=row.get("created_at"),
        )

    async def get_conversation_history(
        self, conversation_id: int, conn: Optional[TypedConnection] = None
    ) -> List[Message]:
        """
        Gets messages for a conversation, ordered and without messages that have been summarized already.

        Args:
            conversation_id: The conversation ID to get messages for
            conn: Optional existing connection for transaction support
        """
        # Try cache first - safe even in transactions for read operations
        cached_messages = cache_storage.get_conversation_messages(conversation_id)
        if cached_messages:
            return self._validate_cached_messages(cached_messages)

        # Acquire connection if not provided
        if conn is None:
            async with self.typed_pool.acquire() as connection:
                return await self._get_conversation_history(conversation_id, connection)
        else:
            return await self._get_conversation_history(conversation_id, conn)

    async def _get_conversation_history(
        self, conversation_id: int, conn: TypedConnection
    ) -> List[Message]:
        """
        Internal method to get conversation history using a specific connection with transaction support.
        """
        # Get all messages for the conversation
        rows = await conn.fetch(
            self.get_query("message.get_conversation_history"),
            conversation_id,
        )

        messages = []
        for row in rows:
            message_id = row["id"]
            # Get the full message with all related data
            message = await self._get_message(message_id, conn, cache_result=True)
            if message:
                messages.append(message)

        # Cache the results - safe to cache read results even in transactions
        if len(messages) > 0:
            try:
                cache_storage.cache_conversation_messages(conversation_id, messages)
            except Exception as e:
                self.logger.warning(f"Failed to cache conversation messages: {e}")

        return messages

    async def get_messages_by_conversation_id(
        self,
        conversation_id: int,
        limit: int,
        offset: int,
        conn: Optional[TypedConnection] = None,
    ) -> List[Message]:
        """
        Gets messages for a conversation by conversation_id with pagination.

        Args:
            conversation_id: The conversation ID to get messages for
            limit: Maximum number of messages to return
            offset: Number of messages to skip
            conn: Optional existing connection for transaction support
        """
        # Check cache first - safe even in transactions for read operations
        cached_messages = cache_storage.get_messages_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_messages:
            return cached_messages

        # Acquire connection if not provided
        if conn is None:
            async with self.typed_pool.acquire() as connection:
                return await self._get_messages_by_conversation_id(
                    conversation_id, limit, offset, connection
                )
        else:
            return await self._get_messages_by_conversation_id(
                conversation_id, limit, offset, conn
            )

    async def _get_messages_by_conversation_id(
        self, conversation_id: int, limit: int, offset: int, conn: TypedConnection
    ) -> List[Message]:
        """
        Internal method to get paginated messages using a specific connection.
        """
        rows = await conn.fetch(
            self.get_query("message.get_by_conversation_id"),
            conversation_id,
            limit,
            offset,
        )

        messages = []
        for row in rows:
            message_id = row["id"]
            # Get the full message with all related data
            message = await self._get_message(message_id, conn, cache_result=True)
            if message:
                messages.append(message)

        # Cache results - safe to cache read results even in transactions
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
        """Helper method to insert message contents using MessageContentStorage."""

        for content in contents:
            # Set the message_id on the content before inserting
            content.message_id = message_id
            await self.message_content_storage.add_content(content=content, conn=conn)

    async def _insert_tool_calls(
        self, conn: TypedConnection, message_id: int, tool_calls: List[ToolCall]
    ) -> None:
        """Helper method to insert tool calls using ToolCallStorage."""

        for tool_call in tool_calls:
            # Set message_id if not already set (for streaming tool_calls)
            if not tool_call.message_id:
                tool_call.message_id = message_id
            await self.tool_call_storage.add_tool_call(tool_call=tool_call, conn=conn)

    async def _insert_thoughts(
        self, conn: TypedConnection, message_id: int, thoughts: List[Thought]
    ) -> None:
        """Helper method to insert thoughts using ThoughtStorage."""

        for thought in thoughts:
            thought_obj = Thought(
                message_id=message_id,
                text=thought if isinstance(thought, str) else str(thought),
            )
            await self.thought_storage.add_thought(thought_obj, conn=conn)

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
                    name=tc_data["name"],
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

                # Parse enums and JSON fields
                workflow_type = WorkflowType(
                    analysis_data.get("workflow_type", "UNKNOWN")
                )
                complexity_level = ComplexityLevel(
                    analysis_data.get("complexity_level", "LOW")
                )

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
                    domain_specificity=float(
                        analysis_data.get("domain_specificity", 0.0)
                    ),
                    reusability_potential=float(
                        analysis_data.get("reusability_potential", 0.0)
                    ),
                    confidence=float(analysis_data.get("confidence", 0.0)),
                    response_format=analysis_data.get("response_format"),
                    technical_domain=analysis_data.get("technical_domain"),
                    requires_tools=bool(analysis_data.get("requires_tools", False)),
                    requires_custom_tools=bool(
                        analysis_data.get("requires_custom_tools", False)
                    ),
                    tool_complexity_score=float(
                        analysis_data.get("tool_complexity_score", 0.0)
                    ),
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
