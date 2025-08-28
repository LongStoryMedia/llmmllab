"""
Enhanced pipeline architecture with full LangGraph integration and improved type safety.
Removes all AgentExecutor dependencies and strengthens the abstraction layer.
"""

from abc import ABC, abstractmethod
import hashlib
import logging
from typing import (
    List,
    Any,
    Type,
    Union,
    AsyncIterator,
    Optional,
    Dict,
    Generic,
    TypeVar,
    Protocol,
    runtime_checkable,
    Sequence,
)
from datetime import datetime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    SystemMessage,
    AIMessage,
    HumanMessage,
    FunctionMessage,
    ToolMessage,
    ChatMessage,
    BaseMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    ModelProfile,
    Model,
    ChatResponse,
    LangGraphState,
)

from .helpers import extract_message_text, to_lc_message

logger = logging.getLogger(__name__)

# Type variable for return type
T = TypeVar("T", bound=Union[str, List[List[float]], ChatResponse])


@runtime_checkable
class LangGraphCapable(Protocol):
    """Protocol for pipelines that support LangGraph workflows."""

    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph:
        """Create a LangGraph workflow for this pipeline."""
        ...

    async def agent_node(self, state: LangGraphState) -> Dict[str, Any]:
        """Process agent node in the graph."""
        ...


class BasePipelineCore(ABC, Generic[T]):
    """
    Core pipeline functionality with mandatory LangGraph support.
    """

    model: Model
    profile: ModelProfile
    memory: MemorySaver

    def __init__(self, model: Model, profile: ModelProfile):
        """Initialize the pipeline with model definition and parameters."""
        self.model = model
        self.profile = profile
        self.memory = MemorySaver()
        self.type = Type[T]

    @abstractmethod
    def create_graph(
        self, tools: Optional[List[BaseTool]] = None
    ) -> CompiledStateGraph[LangGraphState, None, LangGraphState, LangGraphState]:
        """Create LangGraph workflow with caching."""
