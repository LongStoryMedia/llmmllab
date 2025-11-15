"""
Generic workflow execution module for streaming CompiledStateGraph outputs.

This module provides reusable workflow execution capabilities that can be used
across different graph types and state models, extracting the streaming logic
from ComposerService into a generic, reusable component.
"""

import json
from enum import StrEnum
from re import M
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Optional,
    cast,
)
from datetime import datetime, timezone

from pydantic import BaseModel

from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.runnables.schema import StreamEvent, EventData, StandardStreamEvent
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

from models import (
    IntentAnalysis,
    MessageContentType,
    MessageRole,
    Message,
    MessageContent,
    ChatResponse,
    ModelProfileType,
    Thought,
    ToolCall,
)

from runner.pipelines.llamacpp.chat import ReasoningAwareAIMessageChunk
from utils.logging import llmmllogger


class StreamingState(StrEnum):
    """Enum for streaming workflow execution states."""

    THINKING = "thinking"
    EXECUTING = "executing"
    RESPONDING = "responding"
    ANALYZING = "analyzing"


class WorkflowExecutor:
    """
    Generic workflow executor for CompiledStateGraph streaming.

    Provides reusable streaming execution capabilities that can handle
    any CompiledStateGraph with any state type, as long as the state
    can be converted to a dictionary format.
    """

    def __init__(
        self, logger: Optional[Any] = None, default_context: str = "workflow_executor"
    ):
        """
        Initialize the workflow executor.

        Args:
            logger: Optional logger instance. If None, uses default llmmllogger
            default_context: Default context string for metadata enrichment
        """
        self.logger = logger or llmmllogger.logger
        self.default_context = default_context

    def create_thread_config(
        self, thread_id: str, additional_config: Optional[Dict[str, Any]] = None
    ) -> RunnableConfig:
        """
        Create a thread configuration for workflow checkpointing.

        Args:
            thread_id: Unique thread identifier for checkpointing
            additional_config: Additional configuration parameters

        Returns:
            RunnableConfig: Configuration for LangGraph execution
        """
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        if additional_config:
            config.setdefault("configurable", {}).update(additional_config)

        return config

    async def stream_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: BaseModel,
        config: Optional[RunnableConfig] = None,
        thread_id: Optional[str] = None,
        enrich_events: bool = True,
        context_name: Optional[str] = None,
    ) -> AsyncIterator[ChatResponse]:
        """
        Execute a compiled workflow with streaming output.

        This method provides generic streaming execution for any CompiledStateGraph
        and state type, with optional event enrichment and error handling.

        Args:
            workflow: CompiledStateGraph to execute
            initial_state: Initial state for workflow execution (dict or convertible object)
            config: Optional RunnableConfig. If None and thread_id provided, creates default config
            thread_id: Thread ID for checkpointing (used if config is None)
            enrich_events: Whether to enrich events with metadata and tool_calls
            context_name: Context name for metadata (defaults to default_context)

        Yields:
            Dict[str, Any]: Stream events from workflow execution
        """
        start_time = datetime.now(timezone.utc)
        try:
            # Prepare state for execution
            if isinstance(initial_state, dict):
                state_dict = initial_state
            else:
                # Assume state has model_dump method (Pydantic-like)
                if hasattr(initial_state, "model_dump"):
                    state_dict = initial_state.model_dump()
                else:
                    raise ValueError(
                        f"State type {type(initial_state)} must be dict or have model_dump/dict method"
                    )

            # Create config if not provided
            if config is None and thread_id is not None:
                config = self.create_thread_config(thread_id)

            state: StreamingState = StreamingState.RESPONDING
            analyses_buffer = ""
            contents_buffer = ""
            tool_calls_timer: Dict[str, Dict[str, datetime]] = {}
            tool_calls: Dict[str, ToolCall] = {}
            thoughts: Dict[str, Thought] = {}
            analyses: Dict[str, IntentAnalysis] = {}

            # Stream workflow events
            async for event in workflow.astream_events(
                state_dict, config=config, version="v2"
            ):
                try:
                    if enrich_events:
                        event = self._enrich_event(
                            event, context_name or self.default_context
                        )

                except Exception as e:
                    self.logger.warning(
                        "Error enriching workflow event",
                        extra={
                            "error": str(e),
                            "event_type": event.get("event", "unknown"),
                        },
                    )

                data = event.get("data", {})
                event_type = event.get("event", "unknown")
                chunk = data.get("chunk")
                output = data.get("output", {})
                event_name = event.get("name", "unknown")
                run_id = event.get("run_id", "unknown")
                new_state = state

                res = ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[],
                        thoughts=[],
                        tool_calls=[],
                        analyses=[],
                    )
                )

                # make linter happy
                assert res.message
                assert res.message.content is not None
                assert res.message.tool_calls is not None
                assert res.message.analyses is not None
                assert res.message.thoughts is not None

                if event_type == "on_chat_model_start" or event_type == "on_llm_start":
                    md = event.get("metadata", {})
                    task = md.get("task", "Primary")
                    if task == ModelProfileType.Analysis.name:
                        new_state = StreamingState.ANALYZING
                elif event_type == "on_chat_model_end" or event_type == "on_llm_end":
                    if state == StreamingState.ANALYZING:
                        analysis_dict = json.loads(analyses_buffer)
                        analysis = IntentAnalysis(**analysis_dict)
                        analyses[run_id] = analysis
                        res.message.content.append(
                            MessageContent(
                                type=MessageContentType.ANALYSIS, text=analyses_buffer
                            )
                        )
                        res.message.analyses.append(analysis)
                        analyses_buffer = ""
                elif (
                    event_type == "on_chat_model_stream"
                    or event_type == "on_llm_stream"
                ) and isinstance(chunk, AIMessage):
                    if state == StreamingState.ANALYZING:
                        for content in self._parse_content(chunk.content):
                            analyses_buffer += content
                    if hasattr(chunk, "reasoning_content"):
                        new_state = StreamingState.THINKING
                        reasoning_chunk = cast(ReasoningAwareAIMessageChunk, chunk)
                        res.message.thoughts.append(
                            Thought(text=reasoning_chunk.reasoning_content)
                        )
                        res.message.content.append(
                            MessageContent(
                                type=MessageContentType.THINKING,
                                text=reasoning_chunk.reasoning_content,
                            )
                        )
                    if chunk.content is not None:
                        new_state = StreamingState.RESPONDING
                        for content in self._parse_content(chunk.content):
                            res.message.content.append(
                                MessageContent(
                                    type=MessageContentType.TEXT, text=content
                                )
                            )
                            contents_buffer += content

                elif (
                    event_type.endswith("_model_end") or event_type.endswith("_llm_end")
                ) and isinstance(output, AIMessage):
                    self.logger.debug(
                        "Model output received",
                        extra={"output_content": str(output.content)},
                    )
                    md = output.response_metadata or {}
                    reason = md.get("finish_reason") or "unknown"
                    res.done = True
                    res.finish_reason = reason  # type: ignore

                elif event_type.endswith("_tool_start"):
                    self.logger.info(
                        "Tool call started",
                        extra={"tool_name": event_name, "run_id": run_id},
                    )
                    tool_calls_timer[run_id] = {"start": datetime.now(timezone.utc)}
                    tool_calls[run_id] = ToolCall(
                        name=event_name,
                        args=data.get("input", {}),
                        execution_id=run_id,
                    )
                elif event_type.endswith("_tool_end") and isinstance(
                    output, ToolMessage
                ):
                    end_time = datetime.now(timezone.utc)
                    start_time_tc = tool_calls_timer.get(run_id, {}).get("start")
                    duration_ms = int(
                        (end_time - start_time_tc).total_seconds() * 1000
                        if start_time_tc
                        else 0
                    )

                    tool_call = tool_calls.get(run_id)
                    if tool_call is None:
                        self.logger.warning(
                            "Tool call end event without matching start",
                            extra={"run_id": run_id, "tool_name": event_name},
                        )
                        tool_call = ToolCall(
                            name=event_name,
                            args=data.get("input", {}),
                            execution_id=run_id,
                        )
                    tool_call.success = True
                    tool_call.result_data = output.model_dump()
                    tool_call.execution_time_ms = duration_ms
                    tool_call.execution_id = run_id
                    tool_calls[run_id] = tool_call
                    res.message.content.append(
                        MessageContent(
                            type=MessageContentType.TOOL_RESULT,
                            text=str(output.content) or "",
                        )
                    )
                    res.message.tool_calls.append(tool_call)

                if new_state != state:
                    if state == StreamingState.THINKING:
                        thoughts[run_id] = Thought(text=thoughts_buffer)
                        thoughts_buffer = ""
                    elif state == StreamingState.ANALYZING:
                        analysis_dict = json.loads(analyses_buffer)
                        analyses[run_id] = IntentAnalysis(**analysis_dict)
                        analyses_buffer = ""
                    state = new_state

                yield res

        except Exception as e:
            self.logger.error(
                "Workflow execution failed", extra={"error": str(e)}, exc_info=True
            )
            end_time = datetime.now(timezone.utc)
            total_duration = (end_time - start_time).total_seconds() * 1000.0
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text="Sorry, I could not complete your request.",
                        )
                    ],
                ),
                done=True,
                finish_reason="error",
                total_duration=total_duration,
            )

        self.logger.info("Workflow execution completed. Producing final output.")
        yield ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text="",
                    )
                ],
                thoughts=list(thoughts.values()),
                tool_calls=list(tool_calls.values()),
                analyses=list(analyses.values()),
            ),
            done=True,
            finish_reason="complete",
            total_duration=(datetime.now(timezone.utc) - start_time).total_seconds()
            * 1000.0,
        )

    def _parse_content(self, content: str | List[str | Dict[str, Any]]) -> List[str]:
        """
        Parse message content into a list of strings.

        Args:
            content: Content which can be a string or list of strings/dicts

        Returns:
            List[str]: Parsed list of string content
        """
        if isinstance(content, str):
            return [content]
        else:
            return [str(c) for c in content]

    def _enrich_event(self, event: StreamEvent, context_name: str) -> StreamEvent:
        """
        Enrich workflow events with additional metadata and tool information.

        Args:
            event: Original event from workflow execution
            context_name: Context name for metadata

        Returns:
            Dict[str, Any]: Enriched event
        """
        if not isinstance(event, dict):
            return event

        data: EventData = event.get("data")

        # Events that carry a full state snapshot expose 'values'; prefer that
        if data and isinstance(data, dict):
            # If state serialization present
            state_values = data.get("values") or data.get("state")
            if state_values and isinstance(state_values, dict):
                # Create a shallow copy to avoid mutating a typed dict structure
                new_data = dict(data)
                updated = False

                # Inject tool_calls if missing
                tc = state_values.get("tool_calls")
                if tc and "tool_calls" not in data:
                    new_data["tool_calls"] = tc
                    updated = True

                # Inject node metadata if available
                node_metadata = state_values.get("node_metadata")
                if node_metadata and "node_metadata" not in data:
                    new_data["node_metadata"] = node_metadata
                    updated = True

                # Apply enriched data if we made changes
                if updated:
                    event["data"] = new_data  # type: ignore

        # Also check if the event itself has node information we can enrich
        event_type = event.get("event", "")

        # Add execution metadata to certain event types for better traceability
        if event_type in [
            "on_chain_start",
            "on_chain_end",
            "on_tool_start",
            "on_tool_end",
        ]:
            if "metadata" not in event:
                event["metadata"] = {}

            # Add timing and context information
            event["metadata"].update(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "workflow_context": context_name,
                }
            )

        return event


# Convenience factory functions
def create_executor(
    logger: Optional[Any] = None, context: str = "workflow_executor"
) -> WorkflowExecutor:
    """
    Create a new WorkflowExecutor instance.

    Args:
        logger: Optional logger instance
        context: Default context name

    Returns:
        WorkflowExecutor: New executor instance
    """
    return WorkflowExecutor(logger=logger, default_context=context)


async def stream_workflow(
    initial_state: BaseModel,
    workflow: CompiledStateGraph,
    thread_id: Optional[str] = None,
    config: Optional[RunnableConfig] = None,
    logger: Optional[Any] = None,
    context: str = "workflow_stream",
) -> AsyncIterator[ChatResponse]:
    """
    Convenience function for streaming workflow execution.

    Args:
        workflow: CompiledStateGraph to execute
        initial_state: Initial state for workflow execution
        thread_id: Thread ID for checkpointing
        config: Optional RunnableConfig
        logger: Optional logger instance
        context: Context name for metadata

    Yields:
        Dict[str, Any]: Stream events from workflow execution
    """
    executor = create_executor(logger=logger, context=context)

    async for event in executor.stream_workflow(
        workflow=workflow,
        initial_state=initial_state,
        config=config,
        thread_id=thread_id,
        context_name=context,
    ):
        yield event
