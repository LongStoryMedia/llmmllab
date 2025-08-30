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

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the pipeline with model definition and parameters."""
        self.model = model
        self.profile = profile
        self.memory = MemorySaver()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState]:
        """Create LangGraph workflow. Must be implemented by subclasses."""

    @abstractmethod
    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
    ) -> PipeType:
        """Process messages and return appropriate response type."""

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


class ChatPipeline(BasePipelineCore, LangGraphCapable):
    """
    Base class for chat pipelines with LangGraph support.
    """

    def __init__(self, model: Model, profile: ModelProfile):
        super().__init__(model, profile)
        self.graph_cache: Dict[
            str,
            CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState],
        ] = {}
        self.llm: Optional[BaseChatModel] = None

    @abstractmethod
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the underlying LLM. Must be implemented by subclasses."""

    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
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

            return ChatResponse(
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

    @abstractmethod
    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
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

    @abstractmethod
    async def process_messages(
        self, messages: List[Message], tools: Optional[List[BaseTool]] = None
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
