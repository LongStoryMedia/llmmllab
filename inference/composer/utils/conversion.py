"""
Bidirectional message conversion utilities with strong tool call typing.
"""

from typing import List, Optional, Union
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)
from utils.logging import llmmllogger
from .extraction import (
    extract_content_from_message,
    _text_to_message_content_list,
)
from .tool_call_types import (
    LangChainToolCall,
    tool_call_request_to_execution_result,
)

MessageInput = Union[
    str, Message, List[Union[str, Message]], List[str], List[Message]
]  # Debug logging for tool calls conversion

logger = llmmllogger.logger.bind(component="message_conversion")


def to_lc_message(message: Message) -> BaseMessage:
    """Convert a Message object to a format suitable for LangChain."""
    # Extract text content as a simple string
    text_content = extract_content_from_message(message)

    if message.role == MessageRole.ASSISTANT:
        return AIMessage(content=text_content)
    elif message.role == MessageRole.USER:
        return HumanMessage(content=text_content)
    elif message.role == MessageRole.SYSTEM:
        return SystemMessage(content=text_content)
    else:
        # Default to HumanMessage for unknown roles to ensure we always return a BaseMessage
        logger.warning(
            f"Unknown message role: {message.role}, defaulting to HumanMessage"
        )
        return HumanMessage(content=text_content)


def from_lc_message(lc_message: BaseMessage) -> Message:
    """Convert a LangChain BaseMessage to a Message object."""

    # Handle LangChain BaseMessage objects
    if isinstance(lc_message, AIMessage):
        role = MessageRole.ASSISTANT
    elif isinstance(lc_message, HumanMessage):
        role = MessageRole.USER
    elif isinstance(lc_message, SystemMessage):
        role = MessageRole.SYSTEM
    elif isinstance(lc_message, ToolMessage):
        # Tool messages are treated as system messages to preserve tool output context
        role = MessageRole.SYSTEM
    else:
        logger.warning(
            f"Unknown LangChain message type: {type(lc_message)}, defaulting to USER"
        )
        role = MessageRole.USER

    # Extract content - handle both string and list content
    if isinstance(lc_message.content, str):
        text_content = lc_message.content
    elif isinstance(lc_message.content, list):
        # For lists, we'll handle structured parsing below - use empty string as fallback
        text_content = ""
    else:
        # Handle other types by converting to string
        text_content = str(lc_message.content) if lc_message.content else ""

    # Attempt structured parsing if lc_message.content is a list of dicts
    structured_contents: List[MessageContent] = []
    raw_content = getattr(lc_message, "content", None)

    if isinstance(raw_content, list):
        for part in raw_content:
            if (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
            ):
                url = part["image_url"].get("url")
                if url:
                    structured_contents.append(
                        MessageContent(type=MessageContentType.IMAGE, url=url)
                    )
            elif isinstance(part, dict) and part.get("type") == "text":
                txt = part.get("text") or ""
                structured_contents.append(
                    MessageContent(type=MessageContentType.TEXT, text=txt)
                )
            elif isinstance(part, str):
                structured_contents.append(
                    MessageContent(type=MessageContentType.TEXT, text=part)
                )

    if not structured_contents:
        # Fallback to single text content
        structured_contents = [
            MessageContent(type=MessageContentType.TEXT, text=text_content)
        ]

    return Message(
        role=role,
        content=structured_contents,
    )


def message_to_langchain_message(msg: Message):
    """DEPRECATED: Use utils.message_conversion.message_to_base_message instead."""
    from utils.message_conversion import message_to_base_message
    return message_to_base_message(msg)


def langchain_message_to_message(
    lc_msg: BaseMessage, conversation_id: Optional[int] = None
) -> Message:
    """
    Convert a BaseMessage object to a Message object.

    Args:
        lc_msg: BaseMessage object to convert
        conversation_id: Optional conversation ID for the Message

    Returns:
        Converted Message object
    """
    # Preserve structured multimodal content instead of collapsing to plain text.
    # BaseMessage.content may be a list of dicts like:
    # [{"type": "image_url", "image_url": {"url": "..."}}, {"type": "text", "text": "..."}]
    # Previous implementation flattened everything, losing image metadata; this breaks vision models.

    raw_content = getattr(lc_msg, "content", [])
    message_contents: List[MessageContent] = []

    if isinstance(raw_content, list):
        for part in raw_content:
            # Dict with image
            if (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
            ):
                url = part["image_url"].get("url")
                if url:
                    message_contents.append(
                        MessageContent(type=MessageContentType.IMAGE, url=url)
                    )
            # Dict with text
            elif isinstance(part, dict) and part.get("type") == "text":
                text_val = part.get("text") or ""
                message_contents.append(
                    MessageContent(type=MessageContentType.TEXT, text=text_val)
                )
            # Raw string part
            elif isinstance(part, str):
                message_contents.append(
                    MessageContent(type=MessageContentType.TEXT, text=part)
                )
            else:
                # Fallback: string representation
                try:
                    repr_text = str(part)
                    if repr_text and repr_text != "None":
                        message_contents.append(
                            MessageContent(type=MessageContentType.TEXT, text=repr_text)
                        )
                except Exception:
                    # Ignore unparsable parts
                    pass
    else:
        # Single (likely text) content
        content_text = str(raw_content) if raw_content else ""
        message_contents.append(
            MessageContent(type=MessageContentType.TEXT, text=content_text)
        )

    # Determine role from message type
    role = MessageRole.USER  # Default
    if hasattr(lc_msg, "type") and lc_msg.type:
        msg_type = lc_msg.type.lower()
        if msg_type == "ai":
            role = MessageRole.ASSISTANT
        elif msg_type == "system":
            role = MessageRole.SYSTEM
        elif msg_type in ("user", "human"):
            role = MessageRole.USER

    # Convert LangChain tool call requests to execution results if present
    tool_execution_results = None
    if isinstance(lc_msg, AIMessage) and hasattr(lc_msg, "tool_calls") and lc_msg.tool_calls:
        logger.debug(
            f"Converting {len(lc_msg.tool_calls)} LangChain tool calls to execution results"
        )
        tool_execution_results = []
        for tc in lc_msg.tool_calls:
            if isinstance(tc, dict) and "name" in tc and "args" in tc:
                result = tool_call_request_to_execution_result(
                    request=LangChainToolCall(
                        name=tc["name"], args=tc["args"], id=tc.get("id")
                    ),
                    success=True,
                    result_data={"status": "completed"},
                )
                tool_execution_results.append(result)

    # Fallback: ensure at least one text item so downstream logic doesn't see empty content
    if not message_contents:
        message_contents.append(MessageContent(type=MessageContentType.TEXT, text=""))

    return Message(
        content=message_contents,
        role=role,
        conversation_id=conversation_id,
        tool_calls=tool_execution_results,
    )


def convert_messages_to_langchain(messages: List[Message]) -> List[BaseMessage]:
    """Convert a list of Message objects to BaseMessage objects."""
    langchain_messages = []

    for msg in messages:
        if hasattr(msg, "content") and hasattr(msg, "role"):
            # Convert from Message to BaseMessage
            langchain_messages.append(message_to_langchain_message(msg))
        else:
            # Assume already in correct format or convert to dict
            langchain_messages.append(msg)

    return langchain_messages


def convert_messages_to_base_langchain(messages: List[Message]) -> List[BaseMessage]:
    """Convert a list of Message objects to LangChain BaseMessage objects."""

    lc_messages = convert_messages_to_langchain(messages)
    base_messages: List[BaseMessage] = []
    for lc_msg in lc_messages:
        # Get the model dump and fix tool_calls for AI messages
        msg_data = lc_msg.model_dump()
        if lc_msg.type == "ai" and msg_data.get("tool_calls") is None:
            msg_data["tool_calls"] = []

        if lc_msg.type == "human":
            base_messages.append(HumanMessage(**msg_data))
        elif lc_msg.type == "ai":
            base_messages.append(AIMessage(**msg_data))
        elif lc_msg.type == "system":
            base_messages.append(SystemMessage(**msg_data))
        else:
            # Fallback to HumanMessage for unknown types
            base_messages.append(HumanMessage(**msg_data))
    return base_messages


def convert_base_langchain_to_messages(
    base_messages: List[BaseMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """
    Convert a list of LangChain BaseMessage objects to Message objects.

    Args:
        base_messages: List of LangChain BaseMessage objects to convert
        conversation_id: Optional conversation ID for all Message objects

    Returns:
        List of converted Message objects
    """
    return [
        from_lc_message(lc_msg).model_copy(update={"conversation_id": conversation_id})
        for lc_msg in base_messages
    ]


def convert_langchain_messages_to_messages(
    lc_messages: List[BaseMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """
    DEPRECATED: Use utils.message_conversion.base_messages_to_messages instead.
    
    Convert a list of BaseMessage objects to Message objects.
    """
    from utils.message_conversion import base_messages_to_messages
    return base_messages_to_messages(lc_messages, conversation_id)


def normalize_message_input(
    input_data: MessageInput, role: MessageRole = MessageRole.USER
) -> List[Message]:
    """
    Normalize various input types to a List[Message].

    Args:
        input_data: Can be str, Message, List[str | Message]

    Returns:
        List[Message]: Normalized message list
    """
    if isinstance(input_data, str):
        # Single string -> single Message
        return [
            Message(
                role=role,
                content=[MessageContent(type=MessageContentType.TEXT, text=input_data)],
            )
        ]
    elif isinstance(input_data, Message):
        # Single Message -> list with one Message
        return [input_data]
    elif isinstance(input_data, list):
        if not input_data:
            return []

        # Coerce each item in the list to a Message object
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append(
                    Message(
                        role=role,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=item)
                        ],
                    )
                )
            elif isinstance(item, Message):
                messages.append(item)
            else:
                # Convert other types to string, then to Message
                messages.append(
                    Message(
                        role=role,
                        content=[
                            MessageContent(type=MessageContentType.TEXT, text=str(item))
                        ],
                    )
                )
        return messages
