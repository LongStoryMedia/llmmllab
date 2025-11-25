"""
Generic workflow execution module for streaming CompiledStateGraph outputs.

This module provides reusable workflow execution capabilities that can be used
across different graph types and state models, extracting the streaming logic
from ComposerService into a generic, reusable component.
"""

import json
from typing import (
    Any,
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
from langchain_core.runnables.schema import StreamEvent, EventData
from langchain_core.messages import AIMessage, ToolMessage

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
    GenerationState,
)

from runner.pipelines.llamacpp.chat import ReasoningAwareAIMessageChunk
from utils.logging import llmmllogger, serialize_event_data


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

    # What are some recent advancements in multimodal ai models? what are some of the best open source multimodal models available? of those, which is most capable of interpreting text in an image? Find 4 articles detailing the research and development behind that article and list out the links, publication dates, authors, and a brief synopsis. find any citations within those articles and read through the citations, then list those out as well with their links, publication dates, authors and synopses.

    async def stream_workflow(
        self,
        workflow: CompiledStateGraph,
        initial_state: BaseModel,
        config: Optional[RunnableConfig] = None,
        thread_id: Optional[str] = None,
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
        from db import storage  # pylint: disable=import-outside-toplevel

        conversation_id = getattr(initial_state, "conversation_id")
        assert conversation_id is not None and isinstance(
            conversation_id, int
        ), "Initial state must have conversation_id"
        msg_store = storage.get_service(storage.message)

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

            state: Optional[GenerationState] = None
            prev_state: Optional[GenerationState] = state
            analyses_buffer = ""
            contents_buffer = ""
            thoughts_buffer = ""
            # Track the run_id associated with the last appended buffers so
            # flushing uses the correct execution id rather than the current
            # event's run_id which can change between events (e.g. tool calls).
            #
            # Reason: streaming chunks for a single model run may be interleaved
            # with other events (tool calls, sub-workflows) that have different
            # `run_id`s. If we flush or key message contents against the
            # current event's `run_id`, we may accidentally associate the
            # buffered text with the wrong execution id, causing the
            # `contents_buffer` to appear empty for the original run. We
            # therefore record the run id at the time we append content.
            last_content_run_id: Optional[str] = None
            last_thoughts_run_id: Optional[str] = None
            last_analyses_run_id: Optional[str] = None
            tool_calls_timer: Dict[str, Dict[str, datetime]] = {}
            tool_calls: Dict[str, ToolCall] = {}
            thoughts: Dict[str, Thought] = {}
            analyses: Dict[str, IntentAnalysis] = {}
            message_contents: Dict[str, MessageContent] = {}
            total_events = 0

            # Stream workflow events
            async for event in workflow.astream_events(
                state_dict, config=config, version="v2"
            ):

                total_events += 1

                data = event.get("data", {})
                event_type = event.get("event", "unknown")
                chunk = data.get("chunk")
                output = data.get("output", {})
                event_name = event.get("name", "unknown")
                run_id = event.get("run_id", "unknown")
                metadata = event.get("metadata", {})
                new_state = state

                res = ChatResponse(
                    done=False,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=[],
                        thoughts=[],
                        tool_calls=[],
                        analyses=[],
                        conversation_id=conversation_id,
                    ),
                    state=state,
                    prev_state=prev_state,
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
                        new_state = GenerationState.ANALYZING
                elif event_type == "on_chat_model_end" or event_type == "on_llm_end":
                    if state == GenerationState.ANALYZING:
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
                    if not metadata.get("checkpoint_ns", "").startswith("tools_agent"):
                        self.logger.debug(
                            f"Skipping checkpoint_ns: {metadata.get('checkpoint_ns')}",
                        )
                        continue

                    if not metadata.get("node_name", "").startswith(
                        "ToolsAgentSubgraph"
                    ):
                        self.logger.debug(
                            f"Skipping : {metadata.get('node_name')}",
                        )
                        continue

                    if state == GenerationState.ANALYZING:
                        for content in self._parse_content(chunk.content):
                            analyses_buffer += content
                            last_analyses_run_id = run_id
                    if hasattr(chunk, "reasoning_content"):
                        new_state = GenerationState.THINKING
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
                        thoughts_buffer += reasoning_chunk.reasoning_content
                    elif chunk.content:
                        new_state = GenerationState.RESPONDING
                        for content in self._parse_content(chunk.content):
                            res.message.content.append(
                                MessageContent(
                                    type=MessageContentType.TEXT, text=content
                                )
                            )
                            contents_buffer += content
                            # remember which run this content belongs to so we
                            # can flush against the correct execution id later
                            last_content_run_id = run_id

                elif (
                    event_type.endswith("_model_end") or event_type.endswith("_llm_end")
                ) and isinstance(output, AIMessage):
                    self.logger.debug(
                        "Model output received",
                        extra={"output_content": str(output.content)},
                    )
                    md = output.response_metadata or {}
                    reason = md.get("finish_reason") or "unknown"
                    if reason == "tool_call":
                        new_state = GenerationState.EXECUTING
                    if reason == "length":
                        self.logger.warn(
                            "Model generation ended due to length",
                            extra={"run_id": run_id},
                        )

                elif event_type.endswith("_tool_start"):
                    self.logger.info(
                        "Tool call started",
                        extra={"tool_name": event_name, "run_id": run_id},
                    )
                    new_state = GenerationState.EXECUTING
                    tool_calls_timer[run_id] = {"start": datetime.now(timezone.utc)}
                    tool_calls[run_id] = ToolCall(
                        name=event_name,
                        args=data.get("input", {}),
                        execution_id=run_id,
                        created_at=datetime.now(timezone.utc),
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
                    tool_call.created_at = datetime.now(timezone.utc)
                    tool_calls[run_id] = tool_call
                    self.logger.debug(
                        "Appending tool call to response",
                        extra={"tool_call": str(tool_call)},
                    )
                    res.message.tool_calls.append(tool_call)

                prev_state = state

                if new_state != state:
                    self.logger.debug(f"State transition: {state} -> {new_state}")
                    if state == GenerationState.THINKING:
                        self.logger.debug(
                            f"Thoughts buffer: {thoughts_buffer}\n{'-'*20}"
                        )
                        thoughts[run_id] = Thought(
                            text=thoughts_buffer,
                            created_at=datetime.now(timezone.utc),
                        )
                        thoughts_buffer = ""
                    elif state == GenerationState.ANALYZING:
                        self.logger.debug(
                            f"Analyses buffer: {analyses_buffer}\n{'-'*20}"
                        )
                        try:
                            analysis_dict = json.loads(analyses_buffer)
                            analyses[last_analyses_run_id or run_id] = IntentAnalysis(
                                **analysis_dict,
                                created_at=datetime.now(timezone.utc),
                            )
                        except Exception:
                            # keep resiliency for malformed analysis
                            self.logger.debug(
                                "Failed to parse analyses_buffer during flush",
                                extra={"analyses_buffer": analyses_buffer},
                            )
                        analyses_buffer = ""
                    elif state == GenerationState.RESPONDING and contents_buffer:
                        self.logger.debug(
                            f"Contents buffer: {contents_buffer}\n{'-'*20}\n"
                        )
                        # Use the run id that we recorded when appending
                        # to the contents buffer. The current event's
                        # `run_id` may refer to a different execution (for
                        # example a tool call) so using that would miss the
                        # original streamed content.
                        keyed_run_id = last_content_run_id or run_id
                        content_block = MessageContent(
                            type=MessageContentType.TEXT,
                            text=contents_buffer,
                            created_at=datetime.now(timezone.utc),
                        )
                        message_contents[keyed_run_id] = content_block
                        contents_buffer = ""
                    state = new_state

                res.state = state
                res.prev_state = prev_state

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
                            created_at=datetime.now(timezone.utc),
                        )
                    ],
                ),
                done=True,
                finish_reason="error",
                total_duration=total_duration,
            )

        if contents_buffer:
            self.logger.debug(
                "Flushing remaining contents_buffer after workflow completion",
                extra={"contents_buffer": contents_buffer},
            )
            # Use the run id that we recorded when appending
            # to the contents buffer. The current event's
            # `run_id` may refer to a different execution (for
            # example a tool call) so using that would miss the
            # original streamed content.
            keyed_run_id = last_content_run_id or run_id
            content_block = MessageContent(
                type=MessageContentType.TEXT,
                text=contents_buffer,
                created_at=datetime.now(timezone.utc),
            )
            message_contents[keyed_run_id] = content_block
            contents_buffer = ""

        if analyses_buffer:
            self.logger.debug(
                "Flushing remaining analyses_buffer after workflow completion",
                extra={"analyses_buffer": analyses_buffer},
            )
            try:
                analysis_dict = json.loads(analyses_buffer)
                analyses[last_analyses_run_id or run_id] = IntentAnalysis(
                    **analysis_dict,
                    created_at=datetime.now(timezone.utc),
                )
            except Exception:
                # keep resiliency for malformed analysis
                self.logger.debug(
                    "Failed to parse analyses_buffer during final flush",
                    extra={"analyses_buffer": analyses_buffer},
                )
            analyses_buffer = ""

        if thoughts_buffer:
            self.logger.debug(
                "Flushing remaining thoughts_buffer after workflow completion",
                extra={"thoughts_buffer": thoughts_buffer},
            )
            thoughts[last_thoughts_run_id or run_id] = Thought(
                text=thoughts_buffer,
                created_at=datetime.now(timezone.utc),
            )
            thoughts_buffer = ""

        self.logger.info("Workflow execution completed. Producing final output.")
        final_message = Message(
            role=MessageRole.ASSISTANT,
            content=list(message_contents.values()),
            thoughts=list(thoughts.values()),
            tool_calls=list(tool_calls.values()),
            analyses=list(analyses.values()),
            conversation_id=conversation_id,
        )
        await msg_store.add_message(final_message)
        final_response = ChatResponse(
            message=final_message,
            done=True,
            finish_reason="complete",
            total_duration=(datetime.now(timezone.utc) - start_time).total_seconds()
            * 1000.0,
        )
        self.logger.debug(f"Final response: {serialize_event_data(final_response)}")
        yield final_response

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
    ):
        yield event
