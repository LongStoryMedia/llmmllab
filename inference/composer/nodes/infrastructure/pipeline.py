"""
Pipeline Node for LangGraph workflows.
Wraps LLM pipeline execution for chat model operations within workflows.
"""

from typing import List, cast

from langchain.tools import BaseTool

from runner import PipelineFactory
from models import (
    ChatResponse,
    LangChainMessage,
    ModelProfileType,
    PipelinePriority,
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)
from models.default_configs import DEFAULT_CIRCUIT_BREAKER_CONFIG
from utils.model_profile import get_model_profile_for_task
from utils.message import extract_message_text
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from composer.utils.state import assemble_context_messages
from composer.utils.conversion import message_to_langchain_message


class PipelineNode:
    """
    Wraps chat-model execution as a graph node.

    Handles both streaming and non-streaming execution based on configuration.
    Retrieves model profiles internally from shared data layer using user_id.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile_type: ModelProfileType,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        stream: bool = False,
    ):
        """
        Initialize pipeline node.

        Args:
            pipeline_factory: Factory for creating pipeline instances
            profile_type: Model profile type (Primary, Analysis, etc.)
            stream: Whether to enable streaming responses
        """
        self.pipeline_factory = pipeline_factory
        self.profile_type = profile_type
        self.stream = stream
        self.priority = priority
        self.logger = composer_logger.logger.bind(component="PipelineNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute pipeline node with grammar-constrained generation.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with response
        """
        # Lazy imports to avoid circular dependency
        from runner import (  # pylint: disable=import-outside-toplevel
            run_pipeline,
            stream_pipeline,
        )

        try:
            assert state.user_config
            assert state.user_id
            assert self.pipeline_factory, "Pipeline factory not configured"

            # Get model profile and circuit breaker config
            mp = await get_model_profile_for_task(
                state.user_config.model_profiles, self.profile_type, state.user_id
            )
            cb = state.user_config.circuit_breaker or DEFAULT_CIRCUIT_BREAKER_CONFIG

            self.logger.info(
                "Executing pipeline node",
                user_id=state.user_id,
                profile_type=self.profile_type.value,
                streaming=self.stream,
                model=mp.model_name if mp else "unknown",
            )

            # Assemble context messages using the new utility function
            context_messages = assemble_context_messages(state)

            # Debug: Log message assembly info
            self.logger.info(
                "Pipeline execution details",
                user_id=state.user_id,
                messages_count=len(context_messages) if context_messages else 0,
                message_preview=str(
                    context_messages[0].content[:100]
                    if context_messages and context_messages[0].content
                    else "No content"
                )[:100],
                tools_count=len(state.available_tools) if state.available_tools else 0,
            )

            # Execute pipeline based on streaming configuration
            with self.pipeline_factory.pipeline(
                mp, ChatResponse, self.priority, cb
            ) as pipe:
                if self.stream:
                    # For streaming: collect all chunks into final response
                    # LangGraph streaming is handled at graph level, not node level
                    final_content = ""
                    tool_calls = []
                    chunk_count = 0

                    self.logger.info(
                        "Starting stream_pipeline execution",
                        user_id=state.user_id,
                        pipeline_type=type(pipe).__name__,
                    )

                    async for chunk in stream_pipeline(
                        context_messages,
                        pipe,
                        cast(List[BaseTool], state.available_tools),
                    ):
                        chunk_count += 1
                        self.logger.info(
                            "Received pipeline chunk",
                            user_id=state.user_id,
                            chunk_num=chunk_count,
                            has_message=bool(chunk.message),
                            chunk_done=chunk.done,
                            content_preview=str(
                                chunk.message.content[:100]
                                if chunk.message and chunk.message.content
                                else "No content"
                            )[:100],
                        )

                        if chunk.message:
                            final_content = extract_message_text(chunk.message)
                            tool_calls.extend(chunk.message.tool_calls or [])
                            if chunk.done:
                                break

                    self.logger.info(
                        "Stream_pipeline completed",
                        user_id=state.user_id,
                        total_chunks=chunk_count,
                        final_content_length=len(final_content),
                        content_preview=(
                            final_content[:100] if final_content else "No content"
                        ),
                    )

                    # If model emitted inline <tool_call> blocks but underlying simple pipeline
                    # did not structure them into tool_calls metadata, extract them now so the
                    # ToolExecutorNode can act on them in the very next node.
                    if not tool_calls and final_content and "<tool_call>" in final_content:
                        import re, json
                        extracted_calls = []
                        pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
                        for match in pattern.finditer(final_content):
                            block = match.group(1).strip().strip('`')
                            try:
                                parsed = json.loads(block)
                                name = parsed.get("name") or parsed.get("tool")
                                args = parsed.get("arguments") or parsed.get("args") or {}
                                if name:
                                    extracted_calls.append({"name": name, "arguments": args})
                            except Exception:  # pragma: no cover - best effort parsing
                                continue
                        if extracted_calls:
                            tool_calls = extracted_calls
                            self.logger.info(
                                "Extracted tool calls from inline markup",
                                user_id=state.user_id,
                                tool_count=len(tool_calls),
                                tools=[c.get("name") for c in tool_calls],
                            )

                    # Create final response from accumulated content
                    response = ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT, text=final_content
                                )
                            ],
                            tool_calls=tool_calls if tool_calls else None,
                        ),
                        finish_reason="stop",
                    )
                else:
                    # For non-streaming: get complete response directly
                    response = await run_pipeline(
                        context_messages,
                        pipe,
                        cast(List[BaseTool], state.available_tools),
                    )

            # Convert response to LangChainMessage and add to state
            if response and response.message:
                # Extract content text safely
                content_text = ""
                if response.message.content:
                    for content_item in response.message.content:
                        if hasattr(content_item, "text") and content_item.text:
                            content_text += content_item.text
                # Preserve tool calls: message_to_langchain_message now copies tool_calls
                assistant_message = message_to_langchain_message(response.message)
            else:
                # Fallback message
                assistant_message = LangChainMessage(
                    type="ai",
                    content="No response generated from pipeline",
                )

            # Add the response to state messages
            state.messages.append(assistant_message)

            return state

        except Exception as e:
            self.logger.error(
                "Pipeline node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
                profile_type=self.profile_type.value,
            )
            raise NodeExecutionError(f"Pipeline execution failed: {e}") from e
