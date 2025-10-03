"""
Web Search Node for LangGraph workflows.
Wraps WebSearchAgent to provide web search operations within workflows.
"""

from typing import Optional

from composer.agents.web_search_agent import WebSearchAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class WebSearchNode:
    """
    LangGraph node wrapper for web search operations.
    
    Provides web search capabilities within LangGraph workflows,
    supporting configurable search depth and result processing.
    """

    def __init__(self):
        """Initialize web search node."""
        self.agent = WebSearchAgent()
        self.logger = composer_logger.logger.bind(component="WebSearchNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Perform web search based on workflow state and user configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with search results
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for web search")

            # Get search query from latest user message or state metadata
            search_query = self._extract_search_query(state)
            if not search_query:
                self.logger.info("No search query found, skipping web search")
                return state

            # Get search configuration from state or user preferences
            search_depth = state.execution_metadata.get("search_depth", "SHALLOW")
            max_results = state.execution_metadata.get("max_search_results", 5)

            self.logger.info(
                "Performing web search",
                user_id=user_id,
                query=search_query[:100],
                search_depth=search_depth,
                max_results=max_results
            )

            # Perform search using the agent
            search_results = await self.agent.perform_search(
                query=search_query,
                user_id=user_id,
                search_depth=search_depth,
                max_results=max_results
            )

            # Store search results in state
            state.search_results = search_results.get("summary", "")
            state.execution_metadata["web_search_results"] = search_results
            state.execution_metadata["search_completed"] = True

            self.logger.info(
                "Web search completed successfully",
                user_id=user_id,
                results_count=search_results.get("result_count", 0),
                content_length=len(state.search_results or "")
            )

            return state

        except Exception as e:
            self.logger.error(
                "Web search node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Web search failed: {str(e)}")
            state.execution_metadata["search_completed"] = False
            return state

    def _extract_search_query(self, state: WorkflowState) -> Optional[str]:
        """
        Extract search query from workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Search query string or None
        """
        # Check if explicit search query is provided in metadata
        explicit_query = state.execution_metadata.get("search_query")
        if explicit_query:
            return explicit_query

        # Extract from latest user message
        for message in reversed(state.messages):
            if hasattr(message, 'type') and message.type == "human":
                if hasattr(message, 'content') and message.content:
                    return str(message.content)

        return None

    async def search_specific_urls(self, state: WorkflowState) -> WorkflowState:
        """
        Extract content from specific URLs provided in state.
        
        Args:
            state: Current workflow state with URLs to extract
            
        Returns:
            Updated workflow state with extracted content
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for URL extraction")

            urls = state.execution_metadata.get("urls_to_extract", [])
            if not urls:
                self.logger.info("No URLs found for extraction")
                return state

            self.logger.info(
                "Extracting content from URLs",
                user_id=user_id,
                url_count=len(urls)
            )

            # Extract content using the agent
            search_query = " ".join([url for url in urls[:3]])  # Use URLs as search terms
            extraction_results = await self.agent.perform_search(
                query=search_query,
                user_id=user_id,
                search_depth="DEEP",
                max_results=len(urls)
            )

            # Store extraction results in state
            state.execution_metadata["url_extraction_results"] = extraction_results
            state.execution_metadata["url_extraction_completed"] = True

            # Combine extracted content for search results
            if extraction_results:
                # extraction_results is a dict from perform_search, get content from it
                content = extraction_results.get("content", "")
                if isinstance(content, list):
                    combined_content = "\n\n".join([
                        item.get("content", "") if isinstance(item, dict) else str(item) 
                        for item in content
                    ])
                else:
                    combined_content = str(content)
                state.search_results = combined_content[:2000]  # Truncate if too long

            self.logger.info(
                "URL extraction completed successfully",
                user_id=user_id,
                successful_extractions=len(extraction_results)
            )

            return state

        except Exception as e:
            self.logger.error(
                "URL extraction failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"URL extraction failed: {str(e)}")
            state.execution_metadata["url_extraction_completed"] = False
            return state