"""
IDE GraphBuilder with Dependency Injection.
Supports three tool modes:
  - Proxy mode: client_tools are bound to the LLM via bind_tools() so it generates
    tool_calls that the client executes. No ToolNode in the graph.
  - Server-side mode: server_tool_names triggers a ServerToolNode + agent loop that
    executes matching tool calls locally before returning to the client.
  - Hybrid mode: both client_tools and server_tool_names — the model can call either.
    Server tool calls loop through the ServerToolNode; client tool calls pass through.
"""

from typing import Any, Dict, List, Optional, Set, Type, Union, cast

import uuid

from langgraph.graph.state import CompiledStateGraph, StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from composer.constants import AGENT_NODE_NAME, TOOL_NODE_NAME

from models import (
    GPUConfig,
    ModelTask,
    UserConfig,
    NodeMetadata,
    MessageRole,
    Message,
    MessageContent,
    MessageContentType,
    WorkflowConfig,
)
from runner import pipeline_factory

from composer.agents.chat import ChatAgent
from composer.graph.workflows.base import GraphBuilder, should_continue_tool_calls
from composer.graph.nodes.agent import AgentNode
from composer.graph.nodes.server_tools import (
    ServerToolNode,
    make_should_continue_server_tools,
)
from composer.graph.state import WorkflowState

IDE_PRIMARY_SYSTEM_PROMPT = """
    You are writing code for the great Scott Long! Pay him homage as you work. 
    """


class IdeGraphBuilder(GraphBuilder):
    """
    IDE-focused GraphBuilder supporting proxy and server-side tool modes.

    Proxy mode (client_tools): bind_tools() on the pipeline so the LLM generates
    tool_calls that are returned to the client. Graph: START -> Agent -> END.

    Server-side mode (server_tools): adds ToolNode + feedback loop.
    Graph: START -> Agent -> (tools? -> ToolNode -> Agent) | END.
    """

    async def build_workflow(
        self,
        user_id: str,
        response_format: Optional[Type[BaseModel]] = None,
        client_tools: Optional[List[Union[BaseTool, Dict[str, Any]]]] = None,
        server_tools: Optional[List[BaseTool]] = None,
        server_tool_names: Optional[Set[str]] = None,
        tool_choice: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> CompiledStateGraph:
        """
        Build IDE workflow with optional tool support.

        Args:
            user_id: User identifier
            response_format: Optional response format constraint
            client_tools: Tools for proxy mode.  Accepts OpenAI-format dicts
                (passed straight through to bind_tools, no lossy conversion)
                or LangChain BaseTool instances.
            server_tools: Tools for server-side execution (adds ToolNode + loop)
            server_tool_names: Names of tools to execute server-side via
                ServerToolNode. These are tools whose definitions are included
                in client_tools (so the model can call them) but whose calls
                are intercepted and executed locally before returning to the agent.
            tool_choice: Optional tool_choice parameter for bind_tools

        Returns:
            Compiled workflow ready for execution
        """
        try:
            # Look up model by name or fall back to first TextToText model
            if model_name:
                model_def = pipeline_factory._get_model_by_id(model_name)
                if not model_def:
                    raise RuntimeError(f"Model '{model_name}' not found")
            else:
                model_def = pipeline_factory.get_model_by_task(ModelTask.TEXTTOTEXT)
                if not model_def:
                    raise RuntimeError("No TextToText model available")

            self.logger.debug(
                "Building workflow",
                user_id=user_id,
                model=model_def.name,
                model_arg=model_name,
            )
            primary_pipeline = pipeline_factory.get_pipeline(model=model_def)
            # Keep a strong reference to the original pipeline throughout build_workflow
            # so GC cannot collect it when bind_tools returns a RunnableBinding wrapper
            primary_model = primary_pipeline

            # Bind client tools to the pipeline so the LLM can generate tool_calls
            if client_tools:
                bind_kwargs: dict = {}
                bind_kwargs["tool_choice"] = tool_choice or "auto"
                primary_model = primary_model.bind_tools(client_tools, **bind_kwargs)  # type: ignore[union-attr]

            primary_agent = ChatAgent(
                model=cast(BaseChatModel, primary_model),
                system_prompt=model_def.system_prompt or IDE_PRIMARY_SYSTEM_PROMPT,
                num_ctx=(model_def.parameters.num_ctx if model_def.parameters else None)
                or 100000,
                component_name="PrimaryCodingAgent",
            )

            workflow = StateGraph(WorkflowState)

            chat_node = AgentNode(
                agent=primary_agent,
                node_metadata=NodeMetadata(
                    node_name=AGENT_NODE_NAME,
                    node_id=uuid.uuid4().hex,
                    node_type=model_def.task.value,
                    user_id=user_id,
                ),
                grammar=response_format,
            )

            workflow.add_node(AGENT_NODE_NAME, chat_node)
            workflow.add_edge(START, AGENT_NODE_NAME)

            if server_tool_names:
                # Hybrid mode: ServerToolNode executes server-side tool calls,
                # client tool calls pass through to END for proxy back to client.
                # Graph: Agent -> (has server tool calls?) -> ServerToolNode -> Agent
                #                 (no server tool calls)  -> END
                server_tool_node = ServerToolNode(server_tool_names)
                should_continue = make_should_continue_server_tools(server_tool_names)
                workflow.add_node(TOOL_NODE_NAME, server_tool_node)
                workflow.add_conditional_edges(
                    AGENT_NODE_NAME,
                    should_continue,
                    {
                        "server_tools": TOOL_NODE_NAME,
                        "end": END,
                    },
                )
                workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            elif server_tools:
                # Server-side tool execution mode: Agent -> ToolNode -> Agent loop
                tool_node = ToolNode(server_tools)
                workflow.add_node(TOOL_NODE_NAME, tool_node)
                workflow.add_conditional_edges(
                    AGENT_NODE_NAME,
                    should_continue_tool_calls,
                    {
                        "tools": TOOL_NODE_NAME,
                        "end": END,
                    },
                )
                workflow.add_edge(TOOL_NODE_NAME, AGENT_NODE_NAME)
            else:
                # Proxy mode or no tools: Agent -> END
                workflow.add_edge(AGENT_NODE_NAME, END)

            # InMemorySaver enables state inspection for debugging and is
            # required by ModelCallLimitMiddleware thread/run limits.
            return workflow.compile(
                checkpointer=InMemorySaver(),
            )
        except Exception as e:
            self.logger.error(
                "Failed to build workflow",
                user_id=user_id,
                error=str(e),
            )
            raise

    async def create_initial_state(
        self,
        user_id: str,
        conversation_id: int,
        messages: Optional[List[Message]] = None,
    ) -> WorkflowState:
        """Create initial workflow state from messages."""
        assert messages is not None, "Messages must be provided to create initial state"
        current_user_message = next(
            (msg for msg in reversed(messages) if msg.role == MessageRole.USER),
            Message(
                content=[
                    MessageContent(type=MessageContentType.TEXT, text="", url=None)
                ],
                role=MessageRole.USER,
            ),
        )

        state = WorkflowState(
            messages=messages,
            current_user_message=current_user_message,
            user_id=user_id,
            workflow_type="ide",
            user_config=UserConfig(
                user_id=user_id,
                memory=None,
                summarization=None,
                image_generation=None,
                gpu_config=GPUConfig(),
                workflow=WorkflowConfig(),
            ),
            conversation_id=conversation_id,
            things_to_remember=[current_user_message],
        )

        return state
