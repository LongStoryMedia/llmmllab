"""
Direct port of Maistro's message.go storage logic to Python with cache integration.
"""

from calendar import c
from typing import List, Optional, Union
import asyncpg
import logging
from datetime import datetime
from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from server.db.cache_storage import cache_storage
from server.db.db_utils import typed_pool

logger = logging.getLogger(__name__)


class MessageStorage:
    def __init__(self, pool: asyncpg.Pool, get_query):
        self.pool = pool
        self.typed_pool = typed_pool(pool)
        self.get_query = get_query
        self.logger = logging.getLogger(__name__)

    async def add_message(
        self, conversation_id: int, role: str, content: Union[str, List[MessageContent]]
    ) -> Optional[int]:
        # Process content to ensure it's in the right format for storage
        content_str = ""
        if isinstance(content, list):
            # Extract text from MessageContent objects
            for content_item in content:
                if isinstance(content_item, MessageContent) and content_item.text:
                    content_str += content_item.text + " "
            content_str = content_str.strip()
        else:
            content_str = str(content)

        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(
                self.get_query("message.add_message"),
                conversation_id,
                role,
                content_str,
            )
            message_id = row["id"] if row and "id" in row else None

            # Cache the new message if successful
            if message_id:
                # Convert string role to MessageRole enum
                try:
                    message_role = MessageRole(role)
                except ValueError:
                    logger.warning(
                        f"Unknown message role: {role}, using system as default"
                    )
                    message_role = MessageRole.SYSTEM

                # Create MessageContent object with the text content
                if isinstance(content, list) and all(
                    isinstance(c, MessageContent) for c in content
                ):
                    message_content = content
                else:
                    message_content = [
                        MessageContent(type=MessageContentType.TEXT, text=content_str)
                    ]

                message = Message(
                    id=message_id,
                    conversation_id=conversation_id,
                    role=message_role,
                    content=message_content,
                    created_at=datetime.now(),
                )
                cache_storage.cache_message(message)

                # Invalidate conversation messages list cache
                cache_storage.invalidate_conversation_messages_cache(conversation_id)

            return message_id

    async def get_message(self, message_id: int) -> Optional[Message]:
        # First try to get from cache
        cached_message = cache_storage.get_message_from_cache(message_id)
        if cached_message:
            return cached_message

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            row = await conn.fetchrow(self.get_query("message.get_message"), message_id)
            if not row:
                return None

            message = Message(**dict(row))

            # Fetch the content for this message
            try:
                c_rows = await conn.fetch(
                    self.get_query("message.get_content"), message_id
                )

                # Process content rows
                if not c_rows:
                    # Ensure there's at least one MessageContent with empty text
                    message.content = [
                        MessageContent(type=MessageContentType.TEXT, text="", url=None)
                    ]
                else:
                    message_contents = []
                    for c_row in c_rows:
                        try:
                            row_dict = dict(c_row)
                            # Ensure content type is valid
                            content_type = row_dict.get("type")
                            if not content_type:
                                content_type = MessageContentType.TEXT

                            content = MessageContent(
                                type=content_type,
                                text=row_dict.get("text_content", ""),
                                url=row_dict.get("url"),
                            )
                            message_contents.append(content)
                        except Exception as inner_e:
                            logger.warning(f"Failed to process content row: {inner_e}")

                    if message_contents:
                        message.content = message_contents
                    else:
                        # Fallback to at least one empty content
                        message.content = [
                            MessageContent(
                                type=MessageContentType.TEXT, text="", url=None
                            )
                        ]
            except Exception as e:
                logger.warning(f"Failed to fetch content for message {message_id}: {e}")
                # Ensure content is at least an empty list to meet validation requirements
                message.content = [
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ]

            # Cache the result for future use
            try:
                cache_storage.cache_message(message)
            except Exception as e:
                logger.warning(f"Failed to cache message {message_id}: {e}")

            return message

    async def get_conversation_history(self, conversation_id: int) -> List[Message]:
        # First try to get from cache
        cached_messages = cache_storage.get_messages_by_conversation_id_from_cache(
            conversation_id
        )
        if cached_messages is not None:
            # Validate cached messages before returning
            validated_messages = []
            for msg in cached_messages:
                # Ensure content is a list
                if not hasattr(msg, "content") or msg.content is None:
                    msg.content = [
                        MessageContent(type=MessageContentType.TEXT, text="")
                    ]
                elif not isinstance(msg.content, list):
                    msg.content = [
                        MessageContent(
                            type=MessageContentType.TEXT, text=str(msg.content)
                        )
                    ]

                # Ensure conversation_id is set
                if not hasattr(msg, "conversation_id") or msg.conversation_id is None:
                    msg.conversation_id = conversation_id

                validated_messages.append(msg)

            return validated_messages

        # If not in cache, get from database
        async with self.typed_pool.acquire() as conn:
            rows = await conn.fetch(
                self.get_query("message.get_conversation_history"), conversation_id
            )
            message_dicts = [dict(row) for row in rows]
            messages: List[Message] = []

            # Cache the results for future use
            if message_dicts:
                for msg in message_dicts:
                    try:
                        c_rows = await conn.fetch(
                            self.get_query("message.get_content"), msg["id"]
                        )
                        msg_content_dicts = [dict(c_row) for c_row in c_rows]

                        # Ensure content is a list of MessageContent objects
                        if not msg_content_dicts:
                            # If no content rows, create a default MessageContent with empty text
                            msg["content"] = [
                                MessageContent(
                                    type=MessageContentType.TEXT, text="", url=None
                                )
                            ]
                        else:
                            msg["content"] = [
                                MessageContent(
                                    type=d.get("type", MessageContentType.TEXT),
                                    text=d.get("text_content", ""),
                                    url=d.get("url"),
                                )
                                for d in msg_content_dicts
                            ]

                        # Ensure conversation_id is set
                        if (
                            "conversation_id" not in msg
                            or msg["conversation_id"] is None
                        ):
                            msg["conversation_id"] = conversation_id

                        # Create the Message object
                        message_obj = Message(**msg)
                        messages.append(message_obj)

                        # Log for debugging
                        logger.debug(
                            f"Created Message with id={message_obj.id}, role={message_obj.role}, "
                            + f"content_type={message_obj.content[0].type if message_obj.content else 'None'}, "
                            + f"conversation_id={message_obj.conversation_id}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create Message object for caching: {e}, msg={msg}"
                        )

                cache_storage.cache_messages_by_conversation_id(
                    conversation_id, messages
                )

            return messages

    async def delete_message(self, message_id: int) -> None:
        # Get the message to find its conversation_id
        message = cache_storage.get_message_from_cache(message_id)
        conversation_id = message.conversation_id if message else None

        async with self.typed_pool.acquire() as conn:
            await conn.execute(self.get_query("message.delete_message"), message_id)

        # Invalidate message cache
        cache_storage.invalidate_message_cache(message_id)

        # Invalidate conversation messages list cache
        if conversation_id:
            cache_storage.invalidate_conversation_messages_cache(conversation_id)
