"""
Message storage service with enhanced support for tool_calls and thoughts.
Handles message persistence, caching, and proper aggregation of related data.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from models.message import Message
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.tool_call import ToolCall
from models.thought import Thought
from models.resource_usage import ResourceUsage
from models.document import Document
from db.cache_storage import cache_storage
from db.thought_storage import ThoughtStorage
from db.tool_call_storage import ToolCallStorage
from db.message_content_storage import MessageContentStorage
from db.document_storage import DocumentStorage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="message_storage")

# --- Inline SQL (previously in db/sql/message/) ---

_ADD_MESSAGE_SQL = text("""
    INSERT INTO messages(conversation_id, role)
    VALUES (:conversation_id, :role)
    RETURNING id
""")

_GET_MESSAGE_SQL = text("""
    SELECT m.id, m.conversation_id, m.role, m.created_at
    FROM messages m
    WHERE m.id = :message_id
""")

_GET_CONVERSATION_HISTORY_SQL = text("""
    SELECT m.id, m.conversation_id, m.role, m.created_at
    FROM messages m
    WHERE m.conversation_id = :conversation_id
      AND m.id NOT IN (
          SELECT CAST(jsonb_array_elements_text(source_ids) AS integer)
          FROM summaries
          WHERE conversation_id = :conversation_id AND level = 1
      )
    ORDER BY m.created_at ASC
""")

_GET_BY_CONVERSATION_ID_SQL = text("""
    SELECT m.id, m.conversation_id, m.role, m.created_at
    FROM messages m
    WHERE m.conversation_id = :conversation_id
    ORDER BY m.created_at ASC
    LIMIT :limit OFFSET :offset
""")

_UPDATE_MESSAGE_SQL = text("""
    UPDATE messages SET role = :role WHERE id = :message_id
""")

_DELETE_MESSAGE_SQL = text("""
    DELETE FROM messages WHERE id = :message_id
""")

_DELETE_MESSAGES_FROM_TIMESTAMP_SQL = text("""
    DELETE FROM messages
    WHERE conversation_id = :conversation_id AND created_at > :created_at
""")

_DELETE_MESSAGE_CONTENTS_SQL = text("""
    DELETE FROM message_contents WHERE message_id = :message_id
""")

_DELETE_TOOL_CALLS_SQL = text("""
    DELETE FROM tool_calls WHERE message_id = :message_id
""")

_DELETE_THOUGHTS_SQL = text("""
    DELETE FROM thoughts WHERE message_id = :message_id
""")

_INSERT_MESSAGE_CONTENT_SQL = text("""
    INSERT INTO message_contents(message_id, type, text_content, url, format, name, created_at)
    VALUES (:message_id, :type, :text_content, :url, :format, :name, COALESCE(:created_at, NOW()))
    RETURNING id
""")

_INSERT_THOUGHT_SQL = text("""
    INSERT INTO thoughts(message_id, text, created_at)
    VALUES (:message_id, :text, COALESCE(:created_at, NOW()))
    RETURNING id
""")

_INSERT_TOOL_CALL_SQL = text("""
    INSERT INTO tool_calls(
        message_id, tool_name, execution_id, success, args, result_data,
        error_message, execution_time_ms, resource_usage, created_at
    ) VALUES (
        :message_id, :tool_name, :execution_id, :success, :args, :result_data,
        :error_message, :execution_time_ms, :resource_usage, COALESCE(:created_at, NOW())
    )
    RETURNING id
""")

_INSERT_DOCUMENT_SQL = text("""
    INSERT INTO documents(
        message_id, user_id, filename, content_type, file_size, content, text_content,
        created_at, updated_at
    ) VALUES (
        :message_id, :user_id, :filename, :content_type, :file_size, :content, :text_content,
        COALESCE(:created_at, NOW()), COALESCE(:updated_at, NOW())
    )
    RETURNING id, created_at, updated_at
""")

_GET_CONTENTS_BY_MESSAGE_SQL = text("""
    SELECT id, message_id, type, text_content, url, format, name, created_at
    FROM message_contents
    WHERE message_id = :message_id
    ORDER BY id
""")

_GET_TOOL_CALLS_BY_MESSAGE_SQL = text("""
    SELECT id, message_id, tool_name, execution_id, success, args, result_data,
           error_message, execution_time_ms, resource_usage, created_at
    FROM tool_calls
    WHERE message_id = :message_id
""")

_GET_THOUGHTS_BY_MESSAGE_SQL = text("""
    SELECT id, message_id, text, created_at
    FROM thoughts
    WHERE message_id = :message_id
""")

_GET_DOCUMENTS_BY_MESSAGE_SQL = text("""
    SELECT id, message_id, user_id, filename, content_type, file_size, content,
           text_content, created_at, updated_at
    FROM documents
    WHERE message_id = :message_id
""")


class MessageStorage:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        thought_storage: ThoughtStorage,
        tool_call_storage: ToolCallStorage,
        message_content_storage: MessageContentStorage,
        document_storage: DocumentStorage,
    ):
        self.session_factory = session_factory
        self.logger = llmmllogger.bind(component="message_storage_instance")

        # Storage service dependencies
        self.thought_storage = thought_storage
        self.tool_call_storage = tool_call_storage
        self.message_content_storage = message_content_storage
        self.document_storage = document_storage

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def add_message(self, message: Message) -> Optional[int]:
        """Add a message with all its related content, tool_calls, and thoughts."""
        if not message.conversation_id:
            raise ValueError("Message must have a conversation_id")

        self.logger.info(f"Adding message to conversation {message.conversation_id}")

        async with self.session_factory() as session:
            try:
                message_id = await self._add_message(message, session)
                await session.commit()
                return message_id
            except Exception:
                await session.rollback()
                raise

    async def update_message(self, message: Message) -> bool:
        """Update an existing message with all its related data."""
        if not message.id:
            raise ValueError("Message must have an id to be updated")
        if not message.conversation_id:
            raise ValueError("Message must have a conversation_id")

        async with self.session_factory() as session:
            try:
                result = await self._update_message(message, session)
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise

    async def get_message(self, message_id: int) -> Optional[Message]:
        """Get a message by ID with all related content, tool_calls, and thoughts."""
        cached_message = cache_storage.get_message_from_cache(message_id)
        if cached_message:
            return cached_message

        async with self.session_factory() as session:
            return await self._get_message(message_id, session, cache_result=True)

    async def get_conversation_history(
        self, conversation_id: int
    ) -> List[Message]:
        """Get messages for a conversation, excluding summarized messages."""
        cached_messages = cache_storage.get_conversation_messages(conversation_id)
        if cached_messages:
            return self._validate_cached_messages(cached_messages)

        async with self.session_factory() as session:
            return await self._get_conversation_history(conversation_id, session)

    async def get_messages_by_conversation_id(
        self, conversation_id: int, limit: int, offset: int
    ) -> List[Message]:
        """Get messages for a conversation with pagination."""
        cached_messages = cache_storage.get_messages_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_messages:
            return cached_messages

        async with self.session_factory() as session:
            return await self._get_messages_by_conversation_id(
                conversation_id, limit, offset, session
            )

    async def delete_message(self, message_id: int) -> None:
        """Delete a message and all its related data."""
        message = await self.get_message(message_id)
        if not message:
            self.logger.warning(
                f"Message {message_id} not found and could not be deleted"
            )
            return

        async with self.session_factory() as session:
            try:
                await session.execute(_DELETE_MESSAGE_SQL, {"message_id": message_id})
                await session.commit()
                self.logger.info(
                    f"Deleted message {message_id} and related data from database"
                )
            except Exception:
                await session.rollback()
                raise

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

    async def delete_all_from_message(self, message: Message) -> int:
        """Delete all messages in a conversation created at or after the specified timestamp."""
        assert message.conversation_id is not None, "Message must have a conversation_id"

        async with self.session_factory() as session:
            try:
                result = await session.execute(
                    _DELETE_MESSAGES_FROM_TIMESTAMP_SQL,
                    {
                        "conversation_id": message.conversation_id,
                        "created_at": message.created_at,
                    },
                )
                # Also update the message itself within the same transaction
                await self._update_message(message, session)
                await session.commit()

                deleted_count = result.rowcount  # type: ignore[attr-defined]
                logger.info(
                    f"Bulk deleted {deleted_count} messages from conversation "
                    f"{message.conversation_id} created > {message.created_at}"
                )
                return deleted_count
            except Exception:
                await session.rollback()
                raise

        # Invalidate conversation messages list cache
        cache_storage.invalidate_conversation_messages_cache(message.conversation_id)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    async def _add_message(
        self, message: Message, session: AsyncSession
    ) -> Optional[int]:
        """Insert a message and all related data within a session."""
        row = await session.execute(
            _ADD_MESSAGE_SQL,
            {"conversation_id": message.conversation_id, "role": message.role},
        )
        message_id = row.mappings().first()["id"]  # type: ignore[index]

        if not message_id:
            self.logger.error("Failed to get message_id after insert")
            return None

        # Insert related data directly (not via sub-storages) to share transaction
        if message.content:
            await self._insert_message_contents(session, message_id, message.content)
        if message.tool_calls:
            await self._insert_tool_calls(session, message_id, message.tool_calls)
        if message.thoughts:
            await self._insert_thoughts(session, message_id, message.thoughts)
        if message.documents:
            await self._insert_documents(session, message_id, message.documents)

        message.id = message_id

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

    async def _update_message(
        self, message: Message, session: AsyncSession
    ) -> bool:
        """Update a message and all related data within a session."""
        if not message.id:
            self.logger.error("Cannot update message without id")
            return False

        try:
            # Update the main message record
            await session.execute(
                _UPDATE_MESSAGE_SQL,
                {"message_id": message.id, "role": message.role},
            )

            # Delete existing related data
            await session.execute(
                _DELETE_MESSAGE_CONTENTS_SQL, {"message_id": message.id}
            )
            await session.execute(
                _DELETE_TOOL_CALLS_SQL, {"message_id": message.id}
            )
            await session.execute(
                _DELETE_THOUGHTS_SQL, {"message_id": message.id}
            )

            # Insert new related data
            if message.content:
                await self._insert_message_contents(
                    session, message.id, message.content
                )
            if message.tool_calls:
                await self._insert_tool_calls(session, message.id, message.tool_calls)
            if message.thoughts:
                await self._insert_thoughts(session, message.id, message.thoughts)

            # Update cache
            if message.conversation_id is not None:
                try:
                    cache_storage.cache_message(message)
                    cache_storage.invalidate_conversation_messages_cache(
                        message.conversation_id
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to update cache for message {message.id}: {e}"
                    )

            self.logger.debug(f"Successfully updated message {message.id}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to update message {message.id}: {e}", exc_info=True
            )
            return False

    async def _get_message(
        self, message_id: int, session: AsyncSession, cache_result: bool = True
    ) -> Optional[Message]:
        """Get a message with all related data from a session."""
        row = await session.execute(
            _GET_MESSAGE_SQL, {"message_id": message_id}
        )
        msg_row = row.mappings().first()
        if not msg_row:
            return None

        message_data = dict(msg_row)

        # Get related data
        contents_rows = await session.execute(
            _GET_CONTENTS_BY_MESSAGE_SQL, {"message_id": message_id}
        )
        message_data["content"] = [
            MessageContent(**dict(r)) for r in contents_rows.mappings()
        ]

        tool_rows = await session.execute(
            _GET_TOOL_CALLS_BY_MESSAGE_SQL, {"message_id": message_id}
        )
        message_data["tool_calls"] = [
            self._parse_tool_call_row(dict(r)) for r in tool_rows.mappings()
        ]

        thought_rows = await session.execute(
            _GET_THOUGHTS_BY_MESSAGE_SQL, {"message_id": message_id}
        )
        message_data["thoughts"] = [
            Thought(**dict(r)) for r in thought_rows.mappings()
        ]

        doc_rows = await session.execute(
            _GET_DOCUMENTS_BY_MESSAGE_SQL, {"message_id": message_id}
        )
        message_data["documents"] = [
            Document(**dict(r)) for r in doc_rows.mappings()
        ]

        message = Message(**message_data)

        if cache_result:
            try:
                cache_storage.cache_message(message)
            except Exception as e:
                self.logger.warning(f"Failed to cache message {message_id}: {e}")

        return message

    async def _get_conversation_history(
        self, conversation_id: int, session: AsyncSession
    ) -> List[Message]:
        """Get conversation history from a session."""
        rows = await session.execute(
            _GET_CONVERSATION_HISTORY_SQL, {"conversation_id": conversation_id}
        )

        messages = []
        for row in rows.mappings():
            message = await self._get_message(row["id"], session, cache_result=True)
            if message:
                messages.append(message)

        if len(messages) > 0:
            try:
                cache_storage.cache_conversation_messages(conversation_id, messages)
            except Exception as e:
                self.logger.warning(f"Failed to cache conversation messages: {e}")

        return messages

    async def _get_messages_by_conversation_id(
        self,
        conversation_id: int,
        limit: int,
        offset: int,
        session: AsyncSession,
    ) -> List[Message]:
        """Get paginated messages from a session."""
        rows = await session.execute(
            _GET_BY_CONVERSATION_ID_SQL,
            {"conversation_id": conversation_id, "limit": limit, "offset": offset},
        )

        messages = []
        for row in rows.mappings():
            message = await self._get_message(row["id"], session, cache_result=True)
            if message:
                messages.append(message)

        if messages:
            try:
                cache_storage.cache_messages_by_conversation_id(
                    conversation_id, messages
                )
            except Exception as e:
                self.logger.warning(f"Failed to cache paginated messages: {e}")

        return messages

    # ------------------------------------------------------------------
    # Insert helpers (operate on a shared session)
    # ------------------------------------------------------------------

    async def _insert_message_contents(
        self, session: AsyncSession, message_id: int, contents: List[MessageContent]
    ) -> None:
        for content in contents:
            if not content.message_id:
                content.message_id = message_id
            await session.execute(
                _INSERT_MESSAGE_CONTENT_SQL,
                {
                    "message_id": message_id,
                    "type": content.type,
                    "text_content": content.text,
                    "url": content.url,
                    "format": content.format,
                    "name": content.name,
                    "created_at": content.created_at,
                },
            )

    async def _insert_tool_calls(
        self, session: AsyncSession, message_id: int, tool_calls: List[ToolCall]
    ) -> None:
        for tc in tool_calls:
            if not tc.message_id:
                tc.message_id = message_id
            args_json = json.dumps(tc.args) if tc.args else None
            result_json = json.dumps(tc.result_data) if tc.result_data else None
            resource_json = (
                json.dumps(tc.resource_usage.model_dump())
                if tc.resource_usage
                else None
            )
            await session.execute(
                _INSERT_TOOL_CALL_SQL,
                {
                    "message_id": message_id,
                    "tool_name": tc.name,
                    "execution_id": tc.execution_id,
                    "success": tc.success,
                    "args": args_json,
                    "result_data": result_json,
                    "error_message": tc.error_message,
                    "execution_time_ms": tc.execution_time_ms,
                    "resource_usage": resource_json,
                    "created_at": tc.created_at,
                },
            )

    async def _insert_thoughts(
        self, session: AsyncSession, message_id: int, thoughts: List[Thought]
    ) -> None:
        for thought in thoughts:
            if not thought.message_id:
                thought.message_id = message_id
            await session.execute(
                _INSERT_THOUGHT_SQL,
                {
                    "message_id": message_id,
                    "text": thought.text,
                    "created_at": thought.created_at,
                },
            )

    async def _insert_documents(
        self, session: AsyncSession, message_id: int, documents: List[Document]
    ) -> None:
        for doc in documents:
            if not doc.message_id:
                doc.message_id = message_id
            await session.execute(
                _INSERT_DOCUMENT_SQL,
                {
                    "message_id": message_id,
                    "user_id": doc.user_id,
                    "filename": doc.filename,
                    "content_type": doc.content_type,
                    "file_size": doc.file_size,
                    "content": doc.content,
                    "text_content": doc.text_content,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at,
                },
            )

    # ------------------------------------------------------------------
    # Parsing helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _parse_tool_call_row(self, row: Dict[str, Any]) -> ToolCall:
        """Parse a tool call row from database into ToolCall object."""
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

        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at).replace(tzinfo=timezone.utc)

        resource_usage = None
        if resource_usage_data:
            try:
                resource_usage = ResourceUsage(**resource_usage_data)
            except Exception as e:
                self.logger.warning(f"Failed to parse resource_usage: {e}")

        return ToolCall(
            message_id=row.get("message_id"),
            name=row.get("tool_name", "UNKNOWN"),
            execution_id=row.get("execution_id"),
            success=row.get("success", False),
            args=args,
            result_data=result_data if result_data else None,
            error_message=row.get("error_message"),
            execution_time_ms=row.get("execution_time_ms"),
            resource_usage=resource_usage,
            created_at=created_at,
        )

    def _validate_cached_messages(self, cached_messages: Any) -> List[Message]:
        """Validate and clean cached messages data."""
        validated_messages = []
        if not isinstance(cached_messages, list):
            cached_messages = [cached_messages]

        for msg in cached_messages:
            if not msg.content:
                msg.content = [MessageContent(type=MessageContentType.TEXT, text="")]
            elif not isinstance(msg.content, list):
                msg.content = [
                    MessageContent(
                        type=MessageContentType.TEXT, text=str(msg.content)
                    )
                ]
            validated_messages.append(msg)

        return validated_messages