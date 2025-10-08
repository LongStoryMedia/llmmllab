"""
Base class for all summarization nodes.

Provides common initialization and error handling patterns while
requiring each subclass to implement its specific summarization logic.
"""

from abc import ABC, abstractmethod

from composer.graph.state import WorkflowState
from composer.agents.summarization_agent import SummarizationAgent
from composer.nodes.base_node import BaseNode
from runner import PipelineFactory


class BaseSummaryNode(BaseNode, ABC):
    """
    Base class for all summarization nodes.

    Provides common initialization and error handling patterns while
    requiring each subclass to implement its specific summarization logic.
    """

    def _initialize_node(self, pipeline_factory: PipelineFactory, **kwargs) -> None:
        """Initialize summarization-specific components."""
        self.agent = SummarizationAgent(pipeline_factory)

    @abstractmethod
    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """Execute the specific summarization operation."""
        raise NotImplementedError
