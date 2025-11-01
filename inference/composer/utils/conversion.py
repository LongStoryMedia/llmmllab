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
    LangChainMessage,
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


def from_lc_message(lc_message: Union[BaseMessage, LangChainMessage]) -> Message:
    """Convert a LangChain message or LangChainMessage to a Message object."""

    # Handle generated LangChainMessage objects (from schemas)
    if isinstance(lc_message, LangChainMessage):
        # Use the type field to determine the role
        message_type = lc_message.type.lower() if lc_message.type else ""

        if message_type in ("ai", "assistant"):
            role = MessageRole.ASSISTANT
        elif message_type in ("human", "user"):
            role = MessageRole.USER
        elif message_type == "system":
            role = MessageRole.SYSTEM
        elif message_type == "tool":
            # Tool messages are treated as system messages to preserve context
            role = MessageRole.SYSTEM
        else:
            logger.warning(
                f"Unknown LangChainMessage type: {lc_message.type}, defaulting to USER"
            )
            role = MessageRole.USER

        # Extract content - LangChainMessage.content can be string or list
        if isinstance(lc_message.content, str):
            text_content = lc_message.content
        elif isinstance(lc_message.content, list):
            # Join list items or convert them to string
            text_content = str(lc_message.content)
        else:
            # Handle other types by converting to string
            text_content = str(lc_message.content) if lc_message.content else ""

    # Handle LangChain core BaseMessage objects
    elif isinstance(lc_message, AIMessage):
        role = MessageRole.ASSISTANT
        text_content = str(lc_message.content) if lc_message.content else ""
    elif isinstance(lc_message, HumanMessage):
        role = MessageRole.USER
        text_content = str(lc_message.content) if lc_message.content else ""
    elif isinstance(lc_message, SystemMessage):
        role = MessageRole.SYSTEM
        text_content = str(lc_message.content) if lc_message.content else ""
    elif isinstance(lc_message, ToolMessage):
        # Tool messages are treated as system messages to preserve tool output context
        role = MessageRole.SYSTEM
        text_content = str(lc_message.content) if lc_message.content else ""
    else:
        logger.warning(
            f"Unknown LangChain message type: {type(lc_message)}, defaulting to USER"
        )
        role = MessageRole.USER

        # Extract content for unknown types
        if hasattr(lc_message, "content"):
            if isinstance(lc_message.content, str):
                text_content = lc_message.content
            elif isinstance(lc_message.content, list):
                text_content = str(lc_message.content)
            else:
                text_content = str(lc_message.content) if lc_message.content else ""
        else:
            text_content = str(lc_message)

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


def message_to_langchain_message(msg: Message) -> LangChainMessage:
    """Convert a Message object to a LangChainMessage object.

    IMPORTANT: Preserve tool_calls so downstream tool processing
    can handle them correctly.
    """
    content = []
    for c in msg.content:
        if c.type == MessageContentType.TEXT:
            content.append({"type": "text", "text": c.text})
        elif c.type == MessageContentType.IMAGE:
            content.append({"type": "image_url", "image_url": {"url": c.url}})
        else:
            # Fallback to text representation for unknown content types
            content.append({"type": "text", "text": str(c)})

    # Determine message type from role
    message_type = "human"  # Default
    if hasattr(msg, "role") and msg.role:
        role_value = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
        if role_value.lower() in ("assistant", "ai", "system"):
            message_type = (
                "ai" if role_value.lower() in ("assistant", "ai") else "system"
            )

    # Convert tool execution results to LangChain format (requests) if needed
    # Note: Our Message.tool_calls are ToolExecutionResult objects (completed executions)
    # LangChain expects tool_calls to be requests, but in practice this conversion
    # is rarely needed since Messages typically don't contain outgoing tool calls
    tool_calls_for_lc = None
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        # This is unusual - typically only AI messages going TO LangChain would have tool_calls
        logger.debug(
            f"Converting {len(msg.tool_calls)} tool execution results to LangChain format"
        )
        tool_calls_for_lc = []
        for tool_result in msg.tool_calls:
            if hasattr(tool_result, "name") and hasattr(tool_result, "args"):
                tool_calls_for_lc.append(
                    {
                        "name": tool_result.name,
                        "args": tool_result.args,
                        "id": getattr(tool_result, "execution_id", None),
                    }
                )

    logger.info(
        "Converting Message to LangChainMessage",
        has_tool_calls=tool_calls_for_lc is not None,
        tool_calls_count=len(tool_calls_for_lc) if tool_calls_for_lc else 0,
        tool_calls_preview=(
            str(tool_calls_for_lc)[:200] if tool_calls_for_lc else "None"
        ),
    )

    langchain_msg = LangChainMessage(
        content=content,
        type=message_type,
        tool_calls=tool_calls_for_lc,
    )

    logger.info(
        "Created LangChainMessage",
        lc_has_tool_calls=langchain_msg.tool_calls is not None,
        lc_tool_calls_count=(
            len(langchain_msg.tool_calls) if langchain_msg.tool_calls else 0
        ),
    )

    return langchain_msg


def langchain_message_to_message(
    lc_msg: LangChainMessage, conversation_id: Optional[int] = None
) -> Message:
    """
    Convert a LangChainMessage object to a Message object.

    Args:
        lc_msg: LangChainMessage object to convert
        conversation_id: Optional conversation ID for the Message

    Returns:
        Converted Message object
    """
    # Preserve structured multimodal content instead of collapsing to plain text.
    # LangChainMessage.content may be a list of dicts like:
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
    if hasattr(lc_msg, "tool_calls") and lc_msg.tool_calls:
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


def convert_messages_to_langchain(messages: List[Message]) -> List[LangChainMessage]:
    """Convert a list of Message objects to LangChainMessage objects."""
    langchain_messages = []

    for msg in messages:
        if hasattr(msg, "content") and hasattr(msg, "role"):
            # Convert from Message to LangChainMessage
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
    lc_messages: List[LangChainMessage], conversation_id: Optional[int] = None
) -> List[Message]:
    """
    Convert a list of LangChainMessage objects to Message objects.

    Args:
        lc_messages: List of LangChainMessage objects to convert
        conversation_id: Optional conversation ID for all Message objects

    Returns:
        List of converted Message objects
    """
    messages = []
    for lc_msg in lc_messages:
        if hasattr(lc_msg, "content") and hasattr(lc_msg, "type"):
            # Convert from LangChainMessage to Message
            messages.append(langchain_message_to_message(lc_msg, conversation_id))
        else:
            # Handle cases where the list might contain Message objects already
            if hasattr(lc_msg, "content") and hasattr(lc_msg, "role"):
                messages.append(lc_msg)
            else:
                # Try to create a Message from whatever we have
                messages.append(
                    Message(
                        content=_text_to_message_content_list(str(lc_msg)),
                        role=MessageRole.USER,
                        conversation_id=conversation_id,
                    )
                )

    return messages


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
