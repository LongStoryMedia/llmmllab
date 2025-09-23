"""
Enhanced pipeline architecture with full LangGraph integration and improved type safety.
Removes all AgentExecutor dependencies and strengthens the abstraction layer.
"""

from abc import ABC, abstractmethod
import logging
from typing import (
    Generic,
    List,
    Any,
    TypeVar,
    Optional,
    Dict,
    Protocol,
    runtime_checkable,
    Union,
)
from datetime import datetime

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from models import (
    Message,
    ModelProfile,
    Model,
    ChatResponse,
    LangGraphState,
    MessageRole,
    MessageContent,
    MessageContentType,
)

from utils.message import to_lc_message


Embeddings = List[List[float]]
PipeReturn = Union[str, Embeddings, ChatResponse]

logger = logging.getLogger(__name__)

# Define the return types clearly
PipeType = TypeVar("PipeType", bound=PipeReturn)


@runtime_checkable
class LangGraphCapable(Protocol):
    """Protocol for pipelines that support LangGraph workflows."""

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create a LangGraph workflow for this pipeline."""
        ...  # pylint: disable=unnecessary-ellipsis

    async def agent_node(self, state: LangGraphState) -> Dict[str, Any]:
        """Process agent node in the graph."""
        ...  # pylint: disable=unnecessary-ellipsis


class BasePipelineCore(ABC, Generic[PipeType]):
    """
    Core pipeline functionality with mandatory LangGraph support.
    Simplified to remove complex generics that were causing type issues.
    """

    # Subclasses should override to restrict supported return types
    allowed_return_types: tuple[type, ...] = (str, ChatResponse, list)
    # Optional subclass default; if provided, will be used when expected_return_type not passed
    default_return_type: Optional[type] = None

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        expected_return_type: Optional[type] = None,
    ):
        """Initialize the pipeline with model definition and parameters."""
        self.model = model
        self.profile = profile
        self.memory = MemorySaver()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        # Resolve expected type and validate against allowed types
        self.expected_return_type: Optional[type] = (
            expected_return_type
            if expected_return_type is not None
            else self.default_return_type
        )
        if self.expected_return_type is not None and not self._allows_return_type(
            self.expected_return_type
        ):
            allowed = ", ".join(t.__name__ for t in self.allowed_return_types)
            raise TypeError(
                f"{self.__class__.__name__} does not support return type "
                f"{getattr(self.expected_return_type, '__name__', str(self.expected_return_type))}. "
                f"Allowed: {allowed}"
            )

    # ---- Return-type enforcement helpers ----
    def _allows_return_type(self, t: type) -> bool:
        return t in self.allowed_return_types

    def allows_return_type(self, t: type) -> bool:
        """Public check for whether this pipeline supports a return type."""
        return self._allows_return_type(t)

    def validate_return_value(self, value: Any) -> None:
        """Validate that a produced value matches the configured expected return type.

        - If expected_return_type is set, enforce it strictly.
        - If not set, ensure the value matches one of the allowed types.
        """

        def _is_embeddings(v: Any) -> bool:
            # A very light structural check: list of lists of floats (or empty)
            if not isinstance(v, list):
                return False
            if not v:
                return True
            return isinstance(v[0], list)

        expected = self.expected_return_type
        if expected is ChatResponse and not isinstance(value, ChatResponse):
            raise TypeError(
                f"Expected ChatResponse from {self.__class__.__name__}, got {type(value).__name__}"
            )
        if expected is str and not isinstance(value, str):
            raise TypeError(
                f"Expected str from {self.__class__.__name__}, got {type(value).__name__}"
            )
        if expected is list and not _is_embeddings(value):
            raise TypeError(
                f"Expected embeddings (List[List[float]]) from {self.__class__.__name__}, got {type(value).__name__}"
            )

        if expected is None:
            # No explicit expectation; permit anything within allowed types
            if isinstance(value, ChatResponse) and not self._allows_return_type(
                ChatResponse
            ):
                raise TypeError(
                    f"{self.__class__.__name__} is not configured to return ChatResponse"
                )
            if isinstance(value, str) and not self._allows_return_type(str):
                raise TypeError(
                    f"{self.__class__.__name__} is not configured to return str"
                )
            if _is_embeddings(value) and not self._allows_return_type(list):
                raise TypeError(
                    f"{self.__class__.__name__} is not configured to return embeddings"
                )

    @abstractmethod
    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState]:
        """Create LangGraph workflow. Must be implemented by subclasses."""

    @abstractmethod
    async def process_messages(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> PipeType:
        """Process messages and return appropriate response type."""

    # ---- Unified public entrypoints ----
    async def run_pipeline(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> PipeType:
        """Unified non-streaming entrypoint (preferred over direct process_messages).

        This wrapper exists so higher layers can depend on a stable API while
        individual pipeline subclasses may evolve their internal implementation.
        """
        return await self.process_messages(
            messages, tools=tools, is_tool_generation=is_tool_generation
        )

    async def prompt(self, text: str | List[str]) -> PipeType:
        """Process a single message and return appropriate response type"""
        if isinstance(text, list):
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=msg,
                        )
                    ],
                )
                for msg in text
            ]
        else:
            messages = [
                Message(
                    role=MessageRole.USER,
                    content=[
                        MessageContent(
                            type=MessageContentType.TEXT,
                            text=text,
                        )
                    ],
                )
            ]
        return await self.process_messages(messages)

    def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            if hasattr(self, "memory"):
                # Clean up memory checkpointer
                self.memory = MemorySaver()

            # Subclasses should override to add specific cleanup
            self._cleanup_resources()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _cleanup_resources(self) -> None:
        """Subclass-specific cleanup. Override as needed."""

    # def get_common_args(self):
    #     """Get common arguments for the pipeline."""
    #     # Calculate optimal context size
    #     size_map = {
    #         "0.5b": 32768,
    #         "1.5b": 32768,
    #         "3b": 32768,
    #         "7b": 32768,
    #         "14b": 65536,
    #         "30b": 131072,
    #         "72b": 131072,
    #     }
    #     model_name = self.model.name.lower()
    #     context_size = next(
    #         (size for key, size in size_map.items() if key in model_name), 32768
    #     )
    # return context_size

    def get_common_args(self):
        """Return common arguments for this pipeline."""
        return {
            "model": self.model.model_dump() if self.model else None,
            "profile": self.profile.model_dump() if self.profile else None,
        }

    # ---- Token-level post-processing hooks ----
    def process_streaming_token(self, content: str) -> Optional["ChatResponse"]:
        """
        Process a single streaming token and return appropriately formatted ChatResponse.

        This method should be overridden by pipelines that need custom token-level processing.
        For example:
        - GPT-OSS pipeline can track harmony channels and route to thinking/content fields
        - Qwen3MoE pipeline can track <think> tags and route accordingly

        Args:
            content: The raw token/content from the LLM

        Returns:
            ChatResponse with content routed to appropriate fields, or None to suppress
        """
        # Default implementation: route all content to message content
        # if not content.strip():
        #     return None

        message = Message(
            role=MessageRole.ASSISTANT,
            content=[MessageContent(type=MessageContentType.TEXT, text=content)],
        )

        return ChatResponse(done=False, message=message, created_at=datetime.now())

    def reset_streaming_state(self) -> None:
        """
        Reset any internal state used for streaming token processing.
        Called at the start of each streaming session.

        Pipelines should override this to reset their token processing state.
        """
        pass

    def finalize_streaming(self) -> Optional["ChatResponse"]:
        """
        Called when streaming is complete to allow final processing.

        Returns:
            Optional final ChatResponse with any remaining content or metadata
        """
        return None

    def _create_streaming_response(self, content: str) -> ChatResponse:
        """Create a streaming response with content."""
        message = Message(
            role=MessageRole.ASSISTANT,
            content=[MessageContent(type=MessageContentType.TEXT, text=content)],
        )
        return ChatResponse(message=message, done=False)

    def _create_thinking_response(self, thinking_content: str) -> ChatResponse:
        """Create a response with thinking content."""
        message = Message(
            role=MessageRole.ASSISTANT,
            thinking=thinking_content,
            content=[],  # Empty content for thinking-only response
        )
        return ChatResponse(message=message, done=False)


class ChatPipeline(BasePipelineCore, LangGraphCapable):
    """
    Base class for chat pipelines with LangGraph support.
    """

    # Chat pipelines must return ChatResponse
    allowed_return_types: tuple[type, ...] = (ChatResponse,)
    default_return_type: Optional[type] = ChatResponse

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=ChatResponse)
        self.graph_cache: Dict[
            str,
            CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState],
        ] = {}
        self.llm: Optional[BaseChatModel] = None

    @abstractmethod
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the underlying LLM. Must be implemented by subclasses."""

    async def process_messages(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> ChatResponse:
        """Process messages and return ChatResponse."""
        # Default implementation - subclasses can override

        if not self.llm:
            self.llm = self._initialize_llm()

        # Convert to LangChain messages
        lc_messages = [to_lc_message(msg) for msg in messages]

        try:
            response = await self.llm.ainvoke(lc_messages)
            response_text = ""
            if response.content:
                if isinstance(response.content, str):
                    response_text = response.content
                elif isinstance(response.content, list):
                    for content in response.content:
                        if isinstance(content, str):
                            response_text += content + " "
                        elif isinstance(content, dict) and "text" in content:
                            response_text += content["text"] + " "

            result = ChatResponse(
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
                created_at=datetime.now(),
                finish_reason="stop",
            )
            # Enforce expected return type contract
            self.validate_return_value(result)
            return result
        except Exception as e:
            self.logger.error(f"Error processing messages: {e}")
            raise

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState]:
        """Create LangGraph workflow with caching."""

        # Create cache key
        tool_signature = hash(tuple(tool.name for tool in (tools or [])))
        cache_key = f"{self.model.id}_{tool_signature}"

        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key]

        # Initialize LLM if needed
        if not self.llm:
            self.llm = self._initialize_llm()

        # Build graph
        workflow = StateGraph(LangGraphState)
        workflow.add_node("agent", self.agent_node)

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
        self.graph_cache[cache_key] = compiled_graph
        return compiled_graph

    async def agent_node(self, state: LangGraphState) -> Dict[str, Any]:
        """Default agent node implementation."""

        if not self.llm:
            self.llm = self._initialize_llm()

        # Check iteration limits
        if state.current_iteration >= state.max_iterations:
            return {
                "messages": [AIMessage(content="Maximum iterations reached.")],
                "current_iteration": state.current_iteration + 1,
            }

        if state.error_count >= 3:
            return {
                "messages": [AIMessage(content="Too many errors encountered.")],
                "error_count": state.error_count + 1,
            }

        try:
            messages = list(state.messages)
            response = await self.llm.ainvoke(messages)  # type: ignore

            return {
                "messages": [response],
                "current_iteration": state.current_iteration + 1,
            }

        except Exception as e:
            self.logger.error(f"Agent node error: {e}")
            return {
                "messages": [AIMessage(content=f"Error: {str(e)[:100]}...")],
                "error_count": state.error_count + 1,
                "current_iteration": state.current_iteration + 1,
            }

    def as_chat_model(self) -> BaseChatModel:
        """Get LangChain chat model interface."""
        if not self.llm:
            self.llm = self._initialize_llm()
        return self.llm

    def _cleanup_resources(self) -> None:
        """Clean up chat pipeline specific resources."""
        try:
            # Clear graph cache
            self.graph_cache.clear()

            # Clean up LLM
            if self.llm:
                if hasattr(self.llm, "cleanup"):
                    self.llm.cleanup()  # type: ignore
                self.llm = None

        except Exception as e:
            self.logger.error(f"Error cleaning up chat pipeline resources: {e}")


class EmbeddingPipeline(BasePipelineCore):
    """Base class for embedding pipelines."""

    # Embedding pipelines must return embeddings (list[list[float]])
    allowed_return_types: tuple[type, ...] = (list,)
    default_return_type: Optional[type] = list

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=list)

    @abstractmethod
    async def process_messages(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> List[List[float]]:
        """Process messages and return embeddings."""
        # To be implemented by subclasses
        raise NotImplementedError("Embedding pipelines must implement process_messages")

    @abstractmethod
    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Embeddings don't need graphs typically."""
        raise NotImplementedError("Embedding pipelines don't use graphs")


class TextPipeline(BasePipelineCore):
    """Base class for text-only pipelines (summarization, etc.)."""

    # Text pipelines must return plain text (str)
    allowed_return_types: tuple[type, ...] = (str,)
    default_return_type: Optional[type] = str

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile, expected_return_type=str)

    @abstractmethod
    async def process_messages(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        is_tool_generation: bool = False,
    ) -> str:
        """Process messages and return text."""
        # To be implemented by subclasses
        raise NotImplementedError("Text pipelines must implement process_messages")

    @abstractmethod
    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Text pipelines may or may not need graphs."""
        raise NotImplementedError(
            "Text pipelines should implement create_graph if needed"
        )
