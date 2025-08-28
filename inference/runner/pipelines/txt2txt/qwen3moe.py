"""
Simplified and improved LangGraph-based implementation for Qwen3 A3B MoE models.
Fixed missing features from legacy implementation and simplified overly complex logic.
"""

import os
import datetime
import logging
from typing import (
    AsyncIterator,
    Generator,
    List,
    Optional,
    Dict,
    Any,
    TypedDict,
    Annotated,
    Sequence,
    cast,
)

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.runnables.config import RunnableConfig

# LangChain core imports
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from transformers import AutoTokenizer
import torch

# Local imports
from models import (
    MessageContent,
    MessageContentType,
    MessageRole,
    Model,
    Message,
    ChatResponse,
    ModelProfile,
)
from ..base import ChatPipeline

logger = logging.getLogger(__name__)


class QwenAgentState(TypedDict):
    """State schema for Qwen LangGraph agents."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    error_count: int
    max_iterations: int
    current_iteration: int


class QwenLangGraphPipe(ChatPipeline):
    """
    Production-ready LangGraph pipeline for Qwen models.
    Replaces AgentExecutor with modern LangGraph architecture.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)

        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Initializing Qwen LangGraph pipeline: {model.id}")

        # Validate model requirements (preserved from legacy)
        if not (model.details and model.model and model.details.parent_model):
            raise ValueError(
                "Model definition for QwenLangGraphPipe must include details for 'gguf_file' and 'parent_model'."
            )

        # Initialize performance tracking (preserved from legacy)
        self._performance_metrics = {
            "total_tokens": 0,
            "total_requests": 0,
            "average_response_time": 0.0,
            "error_count": 0,
        }

        # Initialize core components
        gguf_path = self._get_gguf_path()
        self._validate_gguf_file(gguf_path)
        self._initialize_llm(gguf_path)

        # LangGraph-specific setup
        self.memory = MemorySaver()
        self.graph_cache: Dict[int, CompiledStateGraph] = {}

    def _get_gguf_path(self) -> str:
        """Get GGUF file path (preserved from legacy)."""
        return (
            self.model.details.gguf_file
            if self.model.details.gguf_file
            else self.model.model
        )

    def _validate_gguf_file(self, gguf_path: str) -> None:
        """Enhanced GGUF file validation (preserved from legacy)."""
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        file_size = os.path.getsize(gguf_path)
        if file_size < 1_000_000:  # Less than 1MB is suspicious
            raise ValueError(f"GGUF file too small ({file_size} bytes): {gguf_path}")

        # Test file readability
        try:
            with open(gguf_path, "rb") as f:
                f.read(8)  # Read first 8 bytes
        except Exception as e:
            raise IOError(f"Cannot read GGUF file {gguf_path}: {e}") from e

        self._logger.info(f"GGUF validated: {gguf_path} ({file_size/1_000_000:.2f} MB)")

    def _initialize_llm(self, gguf_path: str) -> None:
        """Initialize LLM with optimized settings (improved from legacy)."""
        context_size = self._get_optimal_context_size()

        self.llm = ChatLlamaCpp(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_batch=512,
            f16_kv=True,
            verbose=os.environ.get("LOG_LEVEL", "warning") == "debug",
            n_parts=-1,
            n_ctx=self.profile.parameters.num_ctx or context_size,
            seed=self.profile.parameters.seed or -1,
            temperature=self.profile.parameters.temperature or 0.7,
            max_tokens=self.profile.parameters.max_tokens or 4096,
            top_p=self.profile.parameters.top_p or 0.8,
            top_k=self.profile.parameters.top_k or 20,
            repeat_penalty=self.profile.parameters.repeat_penalty or 1.05,
            streaming=True,
            stop=self.profile.parameters.stop
            or [
                "<|im_end|>",
                "<|endoftext|>",
                "<|end|>",
            ],
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

    def _get_optimal_context_size(self) -> int:
        """Calculate optimal context size (preserved logic from legacy)."""
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
        context_size = self.profile.parameters.num_ctx

        if not context_size:
            for size, ctx in size_map.items():
                if size in model_name:
                    context_size = ctx
                    break
            else:
                context_size = 32768

        # VRAM-based constraints (preserved from legacy)
        try:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 8:
                    context_size = min(context_size, 16384)
                elif vram_gb < 16:
                    context_size = min(context_size, 32768)
        except Exception:
            pass

        return context_size

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

    # async def run(
    #     self,
    #     messages: List[Message],
    #     *args,
    #     **kwargs,
    # ) -> AsyncIterator[ChatResponse]:
    #     """
    #     Main execution method using LangGraph architecture.
    #     Maintains interface compatibility with legacy implementation.
    #     """
    #     start_time = datetime.datetime.now(datetime.timezone.utc)
    #     self._performance_metrics["total_requests"] += 1
    #     prompt = cast(
    #         Optional[ChatPromptTemplate],
    #         kwargs.get("prompt", args[0] if args else None),
    #     )
    #     tools = cast(
    #         Optional[List[BaseTool]],
    #         kwargs.get("tools", args[1] if len(args) > 1 else None),
    #     )

    #     if not messages:
    #         yield self._create_error_response("No messages provided")
    #         return

    #     try:
    #         # Create graph
    #         graph = self.create_graph(tools)

    #         # Convert messages to LangChain format
    #         lc_messages = [to_lc_message(msg) for msg in messages]

    #         # Add system prompt from ChatPromptTemplate if provided
    #         if prompt:
    #             try:
    #                 # Extract system message from prompt template
    #                 formatted = prompt.format_messages(input="", chat_history=[])
    #                 system_msgs = [
    #                     msg for msg in formatted if isinstance(msg, SystemMessage)
    #                 ]
    #                 if system_msgs:
    #                     lc_messages = [system_msgs[0]] + lc_messages
    #             except Exception:
    #                 # Add default system message if prompt parsing fails
    #                 lc_messages = [
    #                     SystemMessage(content="You are a helpful AI assistant.")
    #                 ] + lc_messages

    #         user_message = messages[-1]
    #         # Create initial state
    #         initial_state: QwenAgentState = {
    #             "messages": lc_messages,
    #             "user_input": extract_message_text(user_message),
    #             "error_count": 0,
    #             "max_iterations": 3,  # Match legacy default
    #             "current_iteration": 0,
    #         }

    #         # Stream execution with simplified processing
    #         token_count = 0
    #         chunk_count = 0
    #         thinking_phase = True

    #         # Generate thread ID for conversation persistence
    #         import hashlib

    #         thread_id = hashlib.md5(str(messages).encode()).hexdigest()[:16]
    #         thread_config: RunnableConfig = {
    #             "configurable": {"thread_id": f"qwen-{thread_id}"}
    #         }

    #         async for event in graph.astream(
    #             initial_state,
    #             config=thread_config,
    #             stream_mode=["messages", "updates", "checkpoints", "tasks", "values"],
    #         ):
    #             try:
    #                 # Handle thinking phase indicator (preserved from legacy)
    #                 if thinking_phase:
    #                     yield self._create_streaming_chunk(
    #                         "🤔 Processing with LangGraph...\n\n"
    #                     )
    #                     thinking_phase = False

    #                 # Process events based on their type
    #                 if isinstance(event, tuple) and len(event) == 2:
    #                     # Handle tuple format (node_name, node_data)
    #                     node_name, node_data = event
    #                     if (
    #                         node_name == "agent"
    #                         and isinstance(node_data, dict)
    #                         and "messages" in node_data
    #                     ):
    #                         for message in node_data["messages"]:
    #                             if isinstance(message, AIMessage) and message.content:
    #                                 content = (
    #                                     message.content
    #                                     if isinstance(message.content, str)
    #                                     else "\n".join(
    #                                         [str(c) for c in message.content]
    #                                     )
    #                                 )
    #                                 token_count += len(content.split())
    #                                 chunk_count += 1
    #                                 yield self._create_streaming_chunk(content)
    #                     elif (
    #                         node_name == "tools"
    #                         and isinstance(node_data, dict)
    #                         and "messages" in node_data
    #                     ):
    #                         # Format tool outputs for tuple format
    #                         tool_outputs = []
    #                         for msg in node_data["messages"]:
    #                             if isinstance(msg, ToolMessage):
    #                                 tool_name = getattr(msg, "name", "unknown_tool")
    #                                 content = str(msg.content)
    #                                 tool_outputs.append(f"🔧 **{tool_name}**")

    #                         if tool_outputs:
    #                             yield self._create_streaming_chunk(
    #                                 "\n".join(tool_outputs)
    #                             )
    #                 elif isinstance(event, dict):
    #                     # Handle dictionary format
    #                     for node_name, node_data in event.items():
    #                         if node_name == "agent" and "messages" in node_data:
    #                             for message in node_data["messages"]:
    #                                 if (
    #                                     isinstance(message, AIMessage)
    #                                     and message.content
    #                                 ):
    #                                     content = (
    #                                         message.content
    #                                         if isinstance(message.content, str)
    #                                         else "\n".join(
    #                                             [str(c) for c in message.content]
    #                                         )
    #                                     )
    #                                     token_count += len(content.split())
    #                                     chunk_count += 1
    #                                     yield self._create_streaming_chunk(content)
    #                         elif node_name == "tools" and "messages" in node_data:
    #                             # Format tool outputs for dict format
    #                             tool_outputs = []
    #                             for msg in node_data["messages"]:
    #                                 if isinstance(msg, ToolMessage):
    #                                     tool_name = getattr(msg, "name", "unknown_tool")
    #                                     content = str(msg.content)
    #                                     tool_outputs.append(f"🔧 **{tool_name}**")

    #                             if tool_outputs:
    #                                 yield self._create_streaming_chunk(
    #                                     "\n".join(tool_outputs)
    #                                 )

    #             except Exception as chunk_error:
    #                 self._logger.warning(f"Chunk processing error: {chunk_error}")
    #                 continue

    #         # Final completion chunk
    #         yield self._create_streaming_chunk("", done=True)

    #     except Exception as e:
    #         self._logger.error(f"LangGraph execution error: {e}", exc_info=True)
    #         self._performance_metrics["error_count"] += 1
    #         yield self._create_error_response(f"Execution error: {str(e)}")

    #     finally:
    #         # Update performance metrics (preserved from legacy)
    #         duration = (
    #             datetime.datetime.now(datetime.timezone.utc) - start_time
    #         ).total_seconds()
    #         self._performance_metrics["total_tokens"] += token_count

    #         # Update rolling average
    #         current_avg = self._performance_metrics["average_response_time"]
    #         request_count = self._performance_metrics["total_requests"]
    #         self._performance_metrics["average_response_time"] = (
    #             current_avg * (request_count - 1) + duration
    #         ) / request_count

    #         self._logger.info(
    #             f"Request completed: {duration:.2f}s, tokens: {token_count}, "
    #             f"chunks: {chunk_count}, avg_time: {self._performance_metrics['average_response_time']:.2f}s"
    #         )

    def _create_error_response(self, error_message: str) -> ChatResponse:
        """Create standardized error response (preserved from legacy)."""
        return ChatResponse(
            done=True,
            message=Message(
                role=MessageRole.ASSISTANT,
                content=[
                    MessageContent(
                        type=MessageContentType.TEXT,
                        text=f"I apologize, but I encountered an error: {error_message}",
                    )
                ],
            ),
            created_at=datetime.datetime.now(datetime.timezone.utc),
            finish_reason="error",
        )

    def get_performance_metrics(self) -> dict:
        """Get performance metrics (preserved from legacy)."""
        metrics = self._performance_metrics.copy()
        total_requests = max(metrics["total_requests"], 1)

        metrics.update(
            {
                "error_rate": metrics["error_count"] / total_requests,
                "tokens_per_request": metrics["total_tokens"] / total_requests,
                "model_info": {
                    "name": self.model.name,
                    "context_size": getattr(self.llm, "n_ctx", "unknown"),
                    "temperature": getattr(self.llm, "temperature", "unknown"),
                },
            }
        )
        return metrics

    def reset_performance_metrics(self) -> None:
        """Reset performance metrics (preserved from legacy)."""
        self._performance_metrics = {
            "total_tokens": 0,
            "total_requests": 0,
            "average_response_time": 0.0,
            "error_count": 0,
        }

    def get_model_info(self) -> dict:
        """Get model information (preserved from legacy)."""
        return {
            "model_name": self.model.name,
            "model_id": self.model.id,
            "model_path": self.model.details.gguf_file if self.model.details else None,
            "context_size": getattr(self.llm, "n_ctx", None),
            "temperature": getattr(self.llm, "temperature", None),
            "max_tokens": getattr(self.llm, "max_tokens", None),
            "gpu_layers": getattr(self.llm, "n_gpu_layers", None),
            "architecture": "LangGraph",
            "performance_metrics": self.get_performance_metrics(),
        }

    def __del__(self) -> None:
        """Enhanced cleanup (preserved from legacy)."""
        try:
            model_name = (
                getattr(self.model, "name", "unknown")
                if hasattr(self, "model")
                else "unknown"
            )

            if hasattr(self, "_performance_metrics"):
                metrics = self._performance_metrics
                self._logger.info(
                    f"QwenLangGraphPipe {model_name} final metrics: "
                    f"requests={metrics['total_requests']}, "
                    f"tokens={metrics['total_tokens']}, "
                    f"avg_time={metrics['average_response_time']:.2f}s, "
                    f"errors={metrics['error_count']}"
                )

            # Clean up resources
            for attr in ["tokenizer", "llm", "memory", "graph_cache"]:
                if hasattr(self, attr):
                    try:
                        delattr(self, attr)
                    except AttributeError:
                        pass

            # CUDA cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.error(f"Error cleaning up QwenLangGraphPipe: {e}")
