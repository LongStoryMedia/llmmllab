"""
Updated streaming.py - Enhanced for both legacy AgentExecutor and modern LangGraph support.
This maintains backward compatibility while adding LangGraph capabilities.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Union
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.schema import StandardStreamEvent

from models import (
    ChatResponse,
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler to capture streaming output from LangChain agents."""

    def __init__(self):
        self.tokens = []
        self.current_step = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when a new token is generated."""
        self.tokens.append(token)

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        self.current_step = f"Using tool: {action.tool}"

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown")
        self.current_step = f"Running {tool_name}..."

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool finishes."""
        self.current_step = "Processing results..."


class UniversalStreamProcessor:
    """
    Universal processor for both AgentExecutor and LangGraph streaming events.
    Automatically detects the event format and processes accordingly.
    """

    def __init__(self, thinking_phase: bool = True):
        self.thinking_phase = thinking_phase
        self.last_content = ""
        self.repetition_count = 0
        self.max_repetitions = 3
        self.repetition_buffer_size = 200
        self.total_content_length = 0
        self.event_count = 0
        self.processor_type = None  # Will be auto-detected

    def _is_repetitive(self, new_content: str) -> bool:
        """Enhanced repetition detection for both legacy and modern systems."""
        combined = self.last_content + new_content
        if len(combined) > self.repetition_buffer_size:
            combined = combined[-self.repetition_buffer_size :]
        self.last_content = combined

        # Pattern-based repetition detection
        if len(combined) > 50:
            for phrase_len in range(15, min(45, len(combined) // 2)):
                phrase = combined[-phrase_len:].strip()
                if phrase and phrase in combined[:-phrase_len]:
                    return True

        # Token-level repetition (important for GGUF models)
        tokens = combined.split()
        if len(tokens) >= 12:
            last_tokens = tokens[-12:]
            for pattern_len in range(3, 7):
                if pattern_len <= len(last_tokens):
                    pattern = last_tokens[-pattern_len:]
                    remainder = last_tokens[:-pattern_len]
                    if (
                        len(remainder) >= pattern_len
                        and remainder[-pattern_len:] == pattern
                    ):
                        return True

        return False

    def process_event(
        self, event: Union[StandardStreamEvent, Dict[str, Any]]
    ) -> Optional[ChatResponse]:
        """
        Universal event processing that works with both AgentExecutor and LangGraph.
        """
        self.event_count += 1

        # Auto-detect event type on first event
        if self.processor_type is None:
            if isinstance(event, dict) and "event" in event:
                self.processor_type = "agent_executor"
            elif isinstance(event, dict) and any(
                key in event for key in ["agent", "tools", "__start__", "__end__"]
            ):
                self.processor_type = "langgraph"
            else:
                self.processor_type = "unknown"

        # Route to appropriate processor
        if self.processor_type == "agent_executor":
            return self._process_agent_executor_event(event)
        elif self.processor_type == "langgraph":
            return self._process_langgraph_event(event)
        else:
            # Fallback processing
            return self._process_generic_event(event)

    def _process_agent_executor_event(
        self, evt: StandardStreamEvent
    ) -> Optional[ChatResponse]:
        """Process legacy AgentExecutor events."""
        event_type = evt["event"]

        if event_type == "on_chat_model_start" or event_type == "on_llm_start":
            if self.thinking_phase:
                self.thinking_phase = False
                return self.create_streaming_chunk("🤔 Analyzing request...\n\n")

        elif event_type == "on_chat_model_stream" or event_type == "on_llm_stream":
            chunk = evt["data"]["chunk"] if "chunk" in evt["data"] else None
            if chunk and hasattr(chunk, "content"):
                content = getattr(chunk, "content", "")
                if content:
                    self.total_content_length += len(content)

                    if self._is_repetitive(content):
                        self.repetition_count += 1
                        if self.repetition_count >= self.max_repetitions:
                            return self.create_streaming_chunk(
                                "\n\n[Terminating repetitive output]", done=True
                            )

                    return self.create_streaming_chunk(content)

        elif event_type == "on_chat_model_end" or event_type == "on_llm_end":
            return self.create_streaming_chunk("\n\n[Model finished]", done=True)

        elif event_type == "on_tool_start":
            tool_name = evt["data"].get("name", "unknown")
            tool_input = evt["data"].get("input", {})

            response = f"\n\n🔧 **Using {tool_name}**\n"
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    truncated_value = str(value)[:100] + (
                        "..." if len(str(value)) > 100 else ""
                    )
                    response += f"   - {key}: {truncated_value}\n"

            return self.create_streaming_chunk(response)

        elif event_type == "on_tool_end":
            tool_output = evt["data"].get("output", "")
            return self.create_streaming_chunk(
                f"✅ **Tool completed successfully**\n{tool_output}\n\n"
            )

        elif event_type == "on_agent_finish":
            output = evt["data"].get("output", "")
            if output:
                return self.create_streaming_chunk(f"\n\n**Final Answer:**\n{output}\n")

        return None

    def _process_langgraph_event(self, event: Dict[str, Any]) -> Optional[ChatResponse]:
        """Process modern LangGraph events."""
        if not isinstance(event, dict):
            return None

        # Handle LangGraph event structure
        for node_name, node_data in event.items():
            if node_name == "agent":
                return self._handle_langgraph_agent(node_data)
            elif node_name == "tools":
                return self._handle_langgraph_tools(node_data)
            elif node_name == "__start__":
                return self._handle_langgraph_start()
            elif node_name == "__end__":
                return self._handle_langgraph_end(node_data)

        return None

    def _handle_langgraph_agent(self, data: Dict[str, Any]) -> Optional[ChatResponse]:
        """Handle LangGraph agent node output."""
        if "messages" in data and data["messages"]:
            from langchain_core.messages import AIMessage

            last_message = data["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.content:
                content = last_message.content

                if content and not self._is_repetitive(content):
                    self.total_content_length += len(content)
                    return self.create_streaming_chunk(content)
                elif self._is_repetitive(content):
                    self.repetition_count += 1
                    if self.repetition_count >= self.max_repetitions:
                        return self.create_streaming_chunk(
                            "\n\n[Stopping repetitive output]", done=True
                        )
        return None

    def _handle_langgraph_tools(self, data: Dict[str, Any]) -> Optional[ChatResponse]:
        """Handle LangGraph tool execution output."""
        if "messages" in data and data["messages"]:
            from langchain_core.messages import ToolMessage

            tool_outputs = []
            for msg in data["messages"]:
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, "name", "unknown_tool")
                    content = str(msg.content)

                    tool_outputs.append(f"🔧 **{tool_name}**")

                    # Truncate long outputs for readability
                    if len(content) > 250:
                        truncated = content[:250] + "..."
                        tool_outputs.append(f"   Result: {truncated}")
                    else:
                        tool_outputs.append(f"   Result: {content}")

                    tool_outputs.append("")  # Add spacing

            if tool_outputs:
                return self.create_streaming_chunk("\n".join(tool_outputs))

        return None

    def _handle_langgraph_start(self) -> Optional[ChatResponse]:
        """Handle LangGraph execution start."""
        if self.thinking_phase:
            self.thinking_phase = False
            return self.create_streaming_chunk("🤔 Processing with LangGraph...\n\n")
        return None

    def _handle_langgraph_end(self, data: Dict[str, Any]) -> Optional[ChatResponse]:
        """Handle LangGraph execution completion."""
        return self.create_streaming_chunk("", done=True)

    def _process_generic_event(self, event: Any) -> Optional[ChatResponse]:
        """Fallback processing for unknown event types."""
        try:
            # Try to extract any text content from the event
            content = None

            if isinstance(event, dict):
                # Look for common content fields
                for field in ["content", "text", "output", "message"]:
                    if field in event:
                        content = str(event[field])
                        break

                # Look for nested content
                if not content and "data" in event:
                    data = event["data"]
                    if isinstance(data, dict):
                        for field in ["content", "text", "output"]:
                            if field in data:
                                content = str(data[field])
                                break

            elif hasattr(event, "content"):
                content = str(event.content)

            elif isinstance(event, str):
                content = event

            if content and content.strip():
                if not self._is_repetitive(content):
                    return self.create_streaming_chunk(content)
                else:
                    self.repetition_count += 1
                    if self.repetition_count >= self.max_repetitions:
                        return self.create_streaming_chunk(
                            "\n[Stopping repetitive output]", done=True
                        )

        except Exception:
            # Silent fallback - don't break streaming for unknown events
            pass

        return None

    def create_streaming_chunk(
        self, text: str, done: bool = False, role: MessageRole = MessageRole.ASSISTANT
    ) -> ChatResponse:
        """Create a streaming chunk as a JSON ChatResponse."""
        message = None
        if text or not done:
            message = Message(
                role=role,
                content=(
                    [MessageContent(type=MessageContentType.TEXT, text=text)]
                    if text
                    else []
                ),
            )

        return ChatResponse(
            done=done,
            message=message,
            created_at=datetime.now(),
            finish_reason="stop" if done else None,
        )

    def create_streaming_string(self, res: ChatResponse) -> str:
        """Create a streaming string representation."""
        return res.model_dump_json() + "\n"

    def create_error_chunk(self, error_message: str) -> ChatResponse:
        """Create an error chunk as a ChatResponse."""
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.OBSERVER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"I apologize, but I encountered an error: {error_message}",
                    )
                ],
            ),
            model="error",
            created_at=datetime.now(),
            finish_reason="error",
        )

    def get_processor_stats(self) -> Dict[str, Any]:
        """Get statistics about the processor's operation."""
        return {
            "processor_type": self.processor_type,
            "event_count": self.event_count,
            "repetition_count": self.repetition_count,
            "total_content_length": self.total_content_length,
            "thinking_phase_completed": not self.thinking_phase,
        }

    def reset_processor(self) -> None:
        """Reset the processor state for a new conversation."""
        self.last_content = ""
        self.repetition_count = 0
        self.total_content_length = 0
        self.event_count = 0
        self.thinking_phase = True
        self.processor_type = None


class EventStreamProcessor:
    """
    Process streaming events from LangChain agents.
    Enhanced with safety mechanisms to prevent infinite loops and repetition.
    """

    def __init__(self, thinking_phase: bool):
        self.thinking_phase = thinking_phase
        # Track the last few tokens to detect repetitions
        self.last_content = ""
        self.repetition_count = 0
        # Set max repetition before terminating
        self.max_repetitions = 3
        # Repetition detection buffer size
        self.repetition_buffer_size = 200
        # Track total generated content length
        self.total_content_length = 0

    def _is_repetitive(self, new_content: str) -> bool:
        """
        Detect repetitive patterns in the generated text.

        Args:
            new_content: The new content to check for repetitions

        Returns:
            bool: True if repetitive pattern detected
        """
        # Update the buffer with new content
        combined = self.last_content + new_content
        if len(combined) > self.repetition_buffer_size:
            combined = combined[-self.repetition_buffer_size :]
        self.last_content = combined

        # Check for repetitive patterns
        # 1. Check for same phrase repeating
        if len(combined) > 50:
            # Look for a phrase of 20-50 chars repeating
            for phrase_len in range(20, min(50, len(combined) // 2)):
                phrase = combined[-phrase_len:]
                if phrase in combined[:-phrase_len]:
                    return True

        # 2. Check for stuttering (same word multiple times)
        words = combined.split()
        if len(words) >= 8:
            last_words = words[-8:]
            unique_words = set(last_words)
            if len(unique_words) <= 3 and len(last_words) >= 6:
                return True

        return False

    def stream_event(self, evt: StandardStreamEvent):
        event_type = evt["event"]

        if event_type == "on_chat_model_start" or event_type == "on_llm_start":
            if self.thinking_phase:
                yield self.create_streaming_chunk(
                    "🤔 Analyzing request..\n\n",
                )
                self.thinking_phase = False

        elif event_type == "on_chat_model_stream" or event_type == "on_llm_stream":
            chunk = evt["data"]["chunk"] if "chunk" in evt["data"] else None
            if chunk and hasattr(chunk, "content"):
                content = getattr(chunk, "content", "")
                if content:
                    # Check for content length limit
                    self.total_content_length += len(content)

                    # Check for repetitive patterns
                    if self._is_repetitive(content):
                        self.repetition_count += 1
                        # After several repetitions, terminate
                        if self.repetition_count >= self.max_repetitions:
                            yield self.create_streaming_chunk(
                                "\n\n[Terminating repetitive output]", done=True
                            )
                            return

                    # Stream the content if we pass checks
                    yield self.create_streaming_chunk(content)

        elif event_type == "on_chat_model_end" or event_type == "on_llm_end":
            yield self.create_streaming_chunk("\n\n[Model finished]", done=True)

        elif event_type == "on_tool_start":
            tool_name = evt["data"].get("name", "unknown")
            tool_input = evt["data"].get("input", {})
            yield self.create_streaming_chunk(f"\n\n🔧 **Using {tool_name}**\n")
            if isinstance(tool_input, dict):
                for key, value in tool_input.items():
                    yield self.create_streaming_chunk(
                        f"   - {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}\n",
                    )

        elif event_type == "on_tool_end":
            tool_output = evt["data"].get("output", "")
            yield self.create_streaming_chunk(
                f"✅ **Tool completed successfully**\n{tool_output}\n\n"
            )

        elif event_type == "on_agent_finish":
            # Final response
            output = evt["data"].get("output", "")
            if output:
                yield self.create_streaming_chunk(f"\n\n**Final Answer:**\n{output}\n")

    def create_streaming_chunk(
        self, text: str, done: bool = False, role: MessageRole = MessageRole.ASSISTANT
    ) -> ChatResponse:
        """
        Create a streaming chunk as a JSON ChatResponse.
        """
        message = None
        if text or not done:
            message = Message(
                role=role,
                content=(
                    [MessageContent(type=MessageContentType.TEXT, text=text)]
                    if text
                    else []
                ),
            )

        return ChatResponse(
            done=done,
            message=message,
            created_at=datetime.now(),
            finish_reason="stop" if done else None,
        )

    def create_streaming_string(self, res: ChatResponse) -> str:
        """
        Create a streaming string representation.
        """
        return res.model_dump_json() + "\n"

    def create_error_chunk(self, error_message: str) -> ChatResponse:
        """
        Create an error chunk as a ChatResponse.
        """
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.OBSERVER,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"I apologize, but I encountered an error: {error_message}",
                    )
                ],
            ),
            model="error",
            created_at=datetime.now(),
            finish_reason="error",
        )


# Factory function to create appropriate processor
def create_stream_processor(
    thinking_phase: bool = True, processor_type: Optional[str] = None
) -> UniversalStreamProcessor:
    """
    Factory function to create the appropriate stream processor.

    Args:
        thinking_phase: Whether to show initial thinking phase
        processor_type: Force a specific processor type ("agent_executor", "langgraph", or None for auto-detect)

    Returns:
        UniversalStreamProcessor instance
    """
    processor = UniversalStreamProcessor(thinking_phase)
    if processor_type:
        processor.processor_type = processor_type
    return processor
