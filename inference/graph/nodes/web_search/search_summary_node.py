"""
Node for synthesizing web search results into coherent responses.

Encapsulates search result processing with synthesis quality assessment,
key point extraction, and source attribution.
"""

from agents.chat import ChatAgent
from graph.state import WorkflowState
from utils import extract_text_from_message
from utils.logging import llmmllogger
from models import NodeMetadata


class SearchSummaryNode:
    """
    Node for synthesizing web search results into coherent responses.

    Encapsulates search result processing with synthesis quality assessment,
    key point extraction, and source attribution.
    """

    def __init__(self, primary_summary_agent: ChatAgent, node_metadata: NodeMetadata):
        self.agent = primary_summary_agent.bind_node_metadata(node_metadata)
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

            # Extract search results and query
            search_results = state.web_search_results
            query = (
                extract_text_from_message(state.current_user_message)
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
                search_results=search_results
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
