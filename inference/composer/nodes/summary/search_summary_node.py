"""
Node for synthesizing web search results into coherent responses.

Encapsulates search result processing with synthesis quality assessment,
key point extraction, and source attribution.
"""

from typing import TYPE_CHECKING

from composer.graph.state import WorkflowState
from composer.utils.extraction import extract_content_from_langchain_message
from utils.logging import llmmllogger


if TYPE_CHECKING:
    from composer.agents.primary_summary_agent import PrimarySummaryAgent


class SearchSummaryNode:
    """
    Node for synthesizing web search results into coherent responses.

    Encapsulates search result processing with synthesis quality assessment,
    key point extraction, and source attribution.
    """

    def __init__(self, primary_summary_agent: "PrimarySummaryAgent"):
        self.agent = primary_summary_agent
        self.logger = llmmllogger.logger.bind(component="SearchSummaryNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Synthesize web search results with metadata and quality assessment.

        Args:
            state: WorkflowState with search results and query information

        Returns:
            Updated state with synthesized search summary and metadata
        """
        try:
            assert state.user_config
            assert state.user_config.web_search
            wsc = state.user_config.web_search

            # Extract search results and query
            search_results = state.web_search_results
            query = state.search_query or (
                extract_content_from_langchain_message(state.current_user_message)
                if state.current_user_message
                else None
            )
            assert query is not None, "Search query must be provided"

            if not search_results:
                self.logger.info("No search results found for summarization")

                return state

            self.logger.info(
                "Performing search result synthesis",
                result_count=len(search_results),
                query=query[:100] if query else "unknown",
            )

            # Generate search synthesis with metadata
            synthesis_result = await self.agent.summarize_search_results(
                search_results=search_results,
                max_length=wsc.max_content_length or 5000,
            )

            # Store comprehensive synthesis in state
            state.search_syntheses.append(synthesis_result)

            self.logger.info(
                "Search result synthesis completed",
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
