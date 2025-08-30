"""
Optimized LangGraph-based implementation for Qwen3 A3B MoE models.
Clean implementation with only essential methods for public API.
"""

import os
import datetime
import logging
from typing import (
    List,
    Optional,
    Dict,
    Any,
    TypedDict,
    Annotated,
    Sequence,
    cast,
    TypeVar,
    Union,
)

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.tools import BaseTool
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
import torch

from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from ..base import BasePipelineCore

T = TypeVar("T", bound=Union[str, ChatResponse])

logger = logging.getLogger(__name__)


class QwenAgentState(TypedDict):
    """State schema for Qwen LangGraph agents."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    error_count: int
    max_iterations: int
    current_iteration: int


class QwenLangGraphPipe(BasePipelineCore[T]):
    """LangGraph pipeline for Qwen models with clean, optimized implementation."""

    def __init__(
        self, model: Model, profile: ModelProfile, return_type: type = ChatResponse
    ):
        super().__init__(model, profile)
        self._return_type = return_type
        self._logger = logging.getLogger(__name__)

        if not (model.details and model.model and model.details.parent_model):
            raise ValueError(
                "Model definition requires 'gguf_file' and 'parent_model' details."
            )

        # Validate GGUF file at initialization
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)

        # LLM and graph initialized lazily in process_messages
        self.llm = None
        self.memory = MemorySaver()
        self.graph_cache: Dict[int, CompiledStateGraph] = {}

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> T:
        """Process messages and return appropriate response type based on generic parameter."""
        from utils.message import to_lc_message

        # Initialize LLM with tools if not already done
        if self.llm is None:
            gguf_path = self._get_gguf_path()
            self._initialize_llm(gguf_path, tools)

        # Convert to LangChain messages
        lc_messages = [to_lc_message(msg) for msg in messages]

        try:
            # Create and run the graph
            graph = self.create_graph(tools)

            # Prepare initial state
            initial_state = QwenAgentState(
                messages=lc_messages,
                user_input="",  # Empty since we're using messages
                current_iteration=0,
                max_iterations=10,  # Reasonable default
                error_count=0,
            )

            # Run the graph
            result = await graph.ainvoke(initial_state)

            # Extract the final message
            if result["messages"]:
                final_message = result["messages"][-1]
                response_text = ""

                if isinstance(final_message, AIMessage):
                    if final_message.content:
                        if isinstance(final_message.content, str):
                            response_text = final_message.content
                        elif isinstance(final_message.content, list):
                            for content in final_message.content:
                                if isinstance(content, str):
                                    response_text += content + " "
                                elif isinstance(content, dict) and "text" in content:
                                    response_text += content["text"] + " "

                response_text = response_text.strip()

                # Return appropriate type based on generic parameter
                if self._return_type == str:
                    return cast(T, response_text)
                else:
                    return cast(
                        T,
                        ChatResponse(
                            done=True,
                            message=Message(
                                role=MessageRole.ASSISTANT,
                                content=[
                                    MessageContent(
                                        type=MessageContentType.TEXT,
                                        text=response_text,
                                    )
                                ],
                            ),
                            created_at=datetime.datetime.now(datetime.timezone.utc),
                            finish_reason="stop",
                        ),
                    )

            # If no response, return error
            error_msg = "No response generated"
            if self._return_type == str:
                return cast(T, error_msg)
            else:
                return cast(
                    T,
                    ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=error_msg,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="error",
                    ),
                )

        except Exception as e:
            self._logger.error(f"Error in process_messages: {e}")
            error_msg = f"Error processing request: {str(e)}"

            if self._return_type == str:
                return cast(T, error_msg)
            else:
                return cast(
                    T,
                    ChatResponse(
                        done=True,
                        message=Message(
                            role=MessageRole.ASSISTANT,
                            content=[
                                MessageContent(
                                    type=MessageContentType.TEXT,
                                    text=error_msg,
                                )
                            ],
                        ),
                        created_at=datetime.datetime.now(datetime.timezone.utc),
                        finish_reason="error",
                    ),
                )

    async def prompt(self, text: str | List[str]) -> T:
        """Process a single message and return appropriate response type."""
        if isinstance(text, list):
            text = " ".join(text)

        # Create a simple user message
        message = Message(
            role=MessageRole.USER,
            content=[
                MessageContent(
                    type=MessageContentType.TEXT,
                    text=text,
                )
            ],
        )

        return await self.process_messages([message])

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph[QwenAgentState, None, QwenAgentState, QwenAgentState]:
        """Create LangGraph with simplified caching."""
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))

        if tool_signature in self.graph_cache:
            return self.graph_cache[tool_signature]

        # Build graph
        workflow = StateGraph(QwenAgentState)
        workflow.add_node("agent", self._agent_node)

        if tools:
            workflow.add_node("tools", ToolNode(tools))
            workflow.add_conditional_edges(
                "agent", tools_condition, {"tools": "tools", END: END}
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        workflow.add_edge(START, "agent")

        compiled_graph = workflow.compile(checkpointer=self.memory)
        self.graph_cache[tool_signature] = compiled_graph
        return compiled_graph

    async def _agent_node(self, state: QwenAgentState) -> Dict[str, Any]:
        """Simplified agent node with essential error handling."""
        # Check limits
        if state["current_iteration"] >= state["max_iterations"]:
            return {
                "messages": [AIMessage(content="I've reached the iteration limit.")],
                "current_iteration": state["current_iteration"] + 1,
            }

        if state["error_count"] >= 3:
            return {
                "messages": [
                    AIMessage(content="I'm experiencing technical difficulties.")
                ],
                "error_count": state["error_count"] + 1,
            }

        try:
            messages = list(state["messages"])
            assert self.llm is not None, "LLM not initialized"
            response = await self.llm.ainvoke(messages)
            return {
                "messages": [response],
                "current_iteration": state["current_iteration"] + 1,
            }
        except Exception as e:
            self._logger.error(f"Agent node error: {e}")
            return {
                "messages": [AIMessage(content=f"Error: {str(e)[:100]}...")],
                "error_count": state["error_count"] + 1,
                "current_iteration": state["current_iteration"] + 1,
            }

    def _get_gguf_path(self) -> str:
        """Get GGUF file path."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Validate GGUF file exists and is readable."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(f"GGUF file too small ({file_size} bytes): {gguf_path}")

        try:
            with open(gguf_path, "rb") as f:
                f.read(8)  # Test readability
        except Exception as e:
            raise IOError(f"Cannot read GGUF file {gguf_path}: {e}") from e

    def _initialize_llm(
        self, gguf_path: str, tools: Optional[List[BaseTool]] = None
    ) -> None:
        """Initialize LLM with optimized settings and optional tools."""
        # Calculate optimal context size
        size_map = {
            "0.5b": 32768,
            "1.5b": 32768,
            "3b": 32768,
            "7b": 32768,
            "14b": 65536,
            "30b": 131072,
            "72b": 131072,
        }
        model_name = self.model.name.lower()
        context_size = next(
            (size for key, size in size_map.items() if key in model_name), 32768
        )

        # Adjust based on available GPU memory if possible
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 8:
                    context_size = min(context_size, 16384)
                elif vram_gb < 16:
                    context_size = min(context_size, 32768)
        except Exception:
            pass

        # Initialize ChatLlamaCpp
        llm_params = {
            "model_path": gguf_path,
            "n_gpu_layers": -1,
            "n_batch": 512,
            "f16_kv": True,
            "verbose": os.environ.get("LOG_LEVEL", "warning") == "debug",
            "n_parts": -1,
            "n_ctx": self.profile.parameters.num_ctx or context_size,
            "seed": self.profile.parameters.seed or -1,
            "temperature": self.profile.parameters.temperature or 0.7,
            "max_tokens": self.profile.parameters.max_tokens or 4096,
            "top_p": self.profile.parameters.top_p or 0.8,
            "top_k": self.profile.parameters.top_k or 20,
            "repeat_penalty": self.profile.parameters.repeat_penalty or 1.05,
            "streaming": True,
            "stop": self.profile.parameters.stop
            or ["<|im_end|>", "<|endoftext|>", "<|end|>"],
            "callback_manager": CallbackManager([StreamingStdOutCallbackHandler()]),
        }

        if tools:
            self.llm = ChatLlamaCpp(**llm_params)
            self.llm = self.llm.bind_tools(tools)
        else:
            self.llm = ChatLlamaCpp(**llm_params)
