"""
Summarization Workflow for Context Extension System.

Implements hierarchical summarization as per context_extension.md architecture,
providing Level 1-3 summarization with master summary consolidation and
integration with vector database memory storage.
"""

from typing import Dict, Any, List, Optional
from enum import StrEnum
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from composer.graph.state import WorkflowState
from composer.agents.summarization_agent import SummarizationAgent
from utils.logging import llmmllogger
from composer.nodes.summary import ConsolidationNode, SearchSummaryNode
from composer.nodes.embeddings import EmbeddingGeneratorNode

from models import Summary, Message
from runner import PipelineFactory


class SummarizationWorkflow:
    """
    Implements hierarchical summarization workflow per context extension architecture.

    Handles Level 1-3 summarization, master summary management, trigger detection,
    content selection, consolidation, and integration with vector database storage.
    """

    def __init__(self, pipeline_factory: PipelineFactory):
        """Initialize summarization workflow components."""
        self.pipeline_factory = pipeline_factory
        self.agent = SummarizationAgent(pipeline_factory)

        # Create discrete summarization nodes
        self.conversation_node = ConsolidationNode(pipeline_factory)
        self.search_node = SearchSummaryNode(pipeline_factory)

        self.logger = llmmllogger.logger.bind(component="SummarizationWorkflow")

    def build_workflow(self) -> CompiledStateGraph:
        """
        Build the summarization workflow graph.

        Returns:
            CompiledStateGraph: Executable summarization workflow
        """
        graph = StateGraph(WorkflowState)

        # Core workflow nodes using discrete summarization nodes
        graph.add_node("conversation_summary", self.conversation_node)
        graph.add_node("search_summary", self.search_node)
        graph.add_node("storage_integration", self._integrate_with_storage)

        # Set entry point to start both summarization processes
        graph.set_entry_point("conversation_summary")

        # Run search summary in parallel after conversation summary
        graph.add_edge("conversation_summary", "search_summary")

        # Both summarization nodes flow to storage integration
        graph.add_edge("search_summary", "storage_integration")

        # Final context update and completion
        graph.add_edge("storage_integration", END)

        return graph.compile()

    # The discrete summarization operations are now handled by specialized nodes:
    # - ConversationSummaryNode handles Level 1 conversation summarization
    # - ConsolidationNode handles Level 2+ hierarchical consolidation
    # - SearchSummaryNode handles search result synthesis
    # - TextSummaryNode handles general text summarization
    #
    # This provides better separation of concerns, reusability, and testability

    async def _integrate_with_storage(self, state: WorkflowState) -> WorkflowState:
        """
        Integrate generated summaries with database and vector storage.
        Uses conversation_id from state and existing summary results.
        """
        try:
            user_id = state.user_id
            assert user_id is not None

            # Lazy import to avoid circular dependency
            from db import storage  # pylint: disable=import-outside-toplevel

            # Use conversation_id from state
            conversation_id = state.conversation_id
            stored_count = 0
            assert conversation_id is not None

            sum_svc = storage.get_service(storage.summary)

            # get current summaries from db
            current_summaries = await sum_svc.get_summaries_for_conversation(
                conversation_id
            )

            for summ in state.summaries:
                # if a summary with same source_ids already exists, skip
                if not any(
                    set(summ.source_ids).intersection(set(cs.source_ids))
                    for cs in current_summaries
                ):
                    await sum_svc.create_summary(summ)
                    stored_count += 1

            self.logger.info(
                "Storage integration completed",
                user_id=user_id,
                summaries_stored=stored_count,
                conversation_id=conversation_id,
            )

            return state

        except Exception as e:
            self.logger.error(
                "Storage integration failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e),
            )
            state.error_details.append(f"Storage integration failed: {str(e)}")
            return state
