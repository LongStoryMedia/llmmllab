"""
Pipeline Node for LangGraph workflows.
Wraps LLM pipeline execution for chat model operations within workflows.
"""

from typing import List, cast, Dict, Any

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
from composer.core.errors import NodeExecutionError
from composer.utils.state import assemble_context_messages
from composer.utils.conversion import message_to_langchain_message
from composer.nodes.base_node import BaseNode


class PipelineNode(BaseNode):
    """
    Wraps chat-model execution as a graph node.

    Handles both streaming and non-streaming execution based on configuration.
    Retrieves model profiles internally from shared data layer using user_id.
    """

    def _initialize_node(
        self,
        pipeline_factory: PipelineFactory = None,
        profile_type: ModelProfileType = None,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        stream: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize pipeline-specific attributes.

        Args:
            pipeline_factory: Factory for creating pipeline instances
            profile_type: Model profile type (Primary, Analysis, etc.)
            priority: Pipeline execution priority
            stream: Whether to enable streaming responses
        """
        self.pipeline_factory = pipeline_factory
        self.profile_type = profile_type
        self.stream = stream
        self.priority = priority

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile_type: ModelProfileType,
        priority: PipelinePriority = PipelinePriority.MEDIUM,
        stream: bool = False,
        node_name: str = None,
    ):
        """
        Initialize pipeline node.

        Args:
            pipeline_factory: Factory for creating pipeline instances
            profile_type: Model profile type (Primary, Analysis, etc.)
            priority: Pipeline execution priority
            stream: Whether to enable streaming responses
            node_name: Optional custom name for this node (for metadata)
        """
        # Use custom name or default based on profile type
        node_name = node_name or f"PipelineNode-{profile_type.value}"
        
        # Initialize base node
        super().__init__(
            node_name=node_name,
            pipeline_factory=pipeline_factory,
            profile_type=profile_type,
            priority=priority,
            stream=stream
        )

    def create_pipeline_metadata(self, pipeline=None) -> Dict[str, Any]:
        """Create pipeline-specific metadata to add to base node metadata."""
        pipeline_metadata = {
            "profile_type": self.profile_type.value,
            "priority": self.priority.value,
            "streaming": self.stream,
        }
        
        # Add pipeline-specific metadata if available
        if pipeline:
            pipeline_metadata.update({
                "pipeline_type": type(pipeline).__name__,
                "model_name": getattr(pipeline.model, 'name', 'unknown') if hasattr(pipeline, 'model') else 'unknown',
                "model_provider": str(getattr(pipeline.model, 'provider', 'unknown')) if hasattr(pipeline, 'model') else 'unknown',
            })
            
        return pipeline_metadata

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
                # Create and store comprehensive node metadata
                pipeline_metadata = self.create_pipeline_metadata(pipe)
                self.store_node_metadata(state, **pipeline_metadata)
                
                if self.stream:
                    # For streaming: collect all chunks into final response (accumulate, do not overwrite)
                    # LangGraph streaming is handled at graph level, not node level
                    final_content = ""
                    tool_calls = []
                    chunk_count = 0

                    self.logger.info(
                        "Starting stream_pipeline execution",
                        user_id=state.user_id,
                        pipeline_type=type(pipe).__name__,
                        node_id=self.node_id,
                    )

                    async for chunk in stream_pipeline(
                        context_messages,
                        pipe,
                        cast(List[BaseTool], state.available_tools),
                    ):
                        chunk_count += 1
                        
                        # Enrich chunk with node metadata for downstream processing
                        self.enrich_with_node_metadata(chunk, state, **pipeline_metadata)
                        
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
                            node_id=self.node_id,
                        )

                        if chunk.message:
                            # Append new text instead of replacing so earlier segments (possibly containing tool call JSON)
                            # are retained for later inline extraction if needed.
                            final_content += extract_message_text(chunk.message)

                            # Debug: Log tool calls in each chunk
                            chunk_tool_calls = chunk.message.tool_calls or []

                            # Always log final chunks and chunks with tool calls
                            if chunk_tool_calls or chunk.done:
                                self.logger.info(
                                    "Streaming chunk details",
                                    user_id=state.user_id,
                                    chunk_tool_calls_count=len(chunk_tool_calls),
                                    chunk_done=chunk.done,
                                    chunk_has_tool_calls=bool(chunk_tool_calls),
                                    tool_calls_preview=(
                                        str(chunk_tool_calls)[:200]
                                        if chunk_tool_calls
                                        else "None"
                                    ),
                                )

                            tool_calls.extend(chunk_tool_calls)
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

                    # Create final response from accumulated content
                    self.logger.info(
                        "Creating final streaming response",
                        user_id=state.user_id,
                        accumulated_tool_calls=len(tool_calls),
                        final_content_length=len(final_content),
                        tool_calls_preview=(
                            str(tool_calls)[:200] if tool_calls else "None"
                        ),
                    )

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
                    self.logger.info(
                        "Starting run_pipeline execution",
                        user_id=state.user_id,
                        node_id=self.node_id,
                    )
                    
                    response = await run_pipeline(
                        context_messages,
                        pipe,
                        cast(List[BaseTool], state.available_tools),
                    )
                    
                    # Enrich response with node metadata
                    self.enrich_with_node_metadata(response, state, **pipeline_metadata)

            # Convert response to LangChainMessage and add to state
            if response and response.message:
                # Extract content text safely
                content_text = ""
                if response.message.content:
                    for content_item in response.message.content:
                        if hasattr(content_item, "text") and content_item.text:
                            content_text += content_item.text
                # Debug: log raw tool_calls on response.message before conversion
                self.logger.info(
                    "PipelineNode: raw response.message tool_calls",
                    user_id=state.user_id,
                    tool_calls_present=bool(
                        getattr(response.message, "tool_calls", None)
                    ),
                    tool_calls=getattr(response.message, "tool_calls", None),
                )
                # Preserve tool calls: message_to_langchain_message now copies tool_calls
                assistant_message = message_to_langchain_message(response.message)
            else:
                # Fallback message
                assistant_message = LangChainMessage(
                    type="ai",
                    content="No response generated from pipeline",
                )

            # Add the response to state messages
            tool_calls = getattr(assistant_message, "tool_calls", None)

            self.logger.info(
                "Appending assistant message",
                user_id=state.user_id,
                tool_calls_present=bool(tool_calls),
                tool_calls=tool_calls,
            )
            state.messages.append(assistant_message)

            # Surface tool calls in state for downstream nodes & streaming events
            state.tool_calls = tool_calls

            return state

        except Exception as e:
            self.logger.error(
                "Pipeline node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
                profile_type=self.profile_type.value,
            )
            raise NodeExecutionError(f"Pipeline execution failed: {e}") from e
