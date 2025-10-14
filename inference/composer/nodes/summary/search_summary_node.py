"""
Node for synthesizing web search results into coherent responses.

Encapsulates search result processing with synthesis quality assessment,
key point extraction, and source attribution.
"""

from typing import TYPE_CHECKING

from composer.graph.state import WorkflowState
from composer.utils.extraction import extract_content_from_langchain_message
from composer.nodes.base_node import BaseNode


if TYPE_CHECKING:
    from composer.agents.summarization_agent import SummarizationAgent


class SearchSummaryNode(BaseNode):
    """
    Node for synthesizing web search results into coherent responses.

    Encapsulates search result processing with synthesis quality assessment,
    key point extraction, and source attribution.
    """

    def __init__(self, summarization_agent: "SummarizationAgent"):
        super().__init__("search_summary", summarization_agent=summarization_agent)
        
    def _initialize_node(self, pipeline_factory=None, **kwargs) -> None:
        """Initialize SearchSummaryNode with dependency injection."""
        self.agent = kwargs.get('summarization_agent')

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Synthesize web search results with metadata and quality assessment.

        Args:
            state: WorkflowState with search results and query information

        Returns:
            Updated state with synthesized search summary and metadata
        """
        try:
            assert state.user_id
            assert state.user_config
            assert state.user_config.web_search
            wsc = state.user_config.web_search
            user_id = state.user_id

            # Extract search results and query
            search_results = state.web_search_results
            query = state.search_query or (
                extract_content_from_langchain_message(state.current_user_message)
                if state.current_user_message
                else None
            )
            assert query is not None, "Search query must be provided"

            if not search_results:
                self.logger.info(
                    "No search results found for summarization", user_id=user_id
                )

                return state

            self.logger.info(
                "Performing search result synthesis",
                user_id=user_id,
                result_count=len(search_results),
                query=query[:100] if query else "unknown",
            )

            # Generate search synthesis with metadata
            synthesis_result = await self.agent.summarize_search_results(
                search_results=search_results,
                user_id=user_id,
                conversation_id=state.conversation_id or 0,
                query=query or "search query",
                max_length=wsc.max_content_length or 5000,
            )

            # Store comprehensive synthesis in state
            state.search_syntheses.append(synthesis_result)

            self.logger.info(
                "Search result synthesis completed",
                user_id=user_id,
                summary_length=len(synthesis_result.synthesis),
                key_points_count=len(synthesis_result.topics),
                source_count=len(synthesis_result.urls),
                synthesis_quality=getattr(
                    synthesis_result, "synthesis_quality", "unknown"
                ),
            )

            return state

        except Exception as e:
            self.logger.error(f"Search synthesis failed: {e}")
            return state
