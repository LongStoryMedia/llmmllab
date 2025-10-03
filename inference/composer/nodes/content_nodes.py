"""
Web Search and Summarization Nodes for LangGraph workflows.
Wraps WebSearchAgent and SummarizationAgent for workflow integration.
"""

from typing import Optional

from composer.agents.web_search_agent import WebSearchAgent
from composer.agents.summarization_agent import SummarizationAgent
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
                content_length=len(state.search_results)
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
            extraction_results = await self.agent.extract_urls_content(
                urls=urls,
                user_id=user_id
            )

            # Store extraction results in state
            state.execution_metadata["url_extraction_results"] = extraction_results
            state.execution_metadata["url_extraction_completed"] = True

            # Combine extracted content for search results
            if extraction_results:
                combined_content = "\n\n".join([
                    result.get("content", "") for result in extraction_results
                ])
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


class SummarizationNode:
    """
    LangGraph node wrapper for content summarization.
    
    Provides summarization capabilities within LangGraph workflows,
    supporting different summary types and styles.
    """

    def __init__(self, pipeline_factory):
        """
        Initialize summarization node.
        
        Args:
            pipeline_factory: Factory for creating summarization pipelines
        """
        self.agent = SummarizationAgent(pipeline_factory)
        self.logger = composer_logger.logger.bind(component="SummarizationNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Summarize content based on workflow state and configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with summarized content
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for summarization")

            # Determine what to summarize
            content_to_summarize = self._get_content_to_summarize(state)
            if not content_to_summarize:
                self.logger.info("No content found for summarization")
                return state

            # Get summarization configuration
            summary_type = state.execution_metadata.get("summary_type", "general")
            summary_style = state.execution_metadata.get("summary_style", "concise")

            self.logger.info(
                "Performing content summarization",
                user_id=user_id,
                content_length=len(content_to_summarize),
                summary_type=summary_type,
                style=summary_style
            )

            # Generate summary using the agent
            summary = await self.agent.summarize_text(
                text=content_to_summarize,
                user_id=user_id,
                summary_type=summary_type,
                style=summary_style
            )

            # Store summary in state
            state.execution_metadata["generated_summary"] = summary
            state.execution_metadata["summarization_completed"] = True

            # Update search results with summary if this was search content
            if state.execution_metadata.get("summarize_search_results", False):
                state.search_results = summary

            self.logger.info(
                "Summarization completed successfully",
                user_id=user_id,
                original_length=len(content_to_summarize),
                summary_length=len(summary)
            )

            return state

        except Exception as e:
            self.logger.error(
                "Summarization node execution failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Summarization failed: {str(e)}")
            state.execution_metadata["summarization_completed"] = False
            return state

    def _get_content_to_summarize(self, state: WorkflowState) -> Optional[str]:
        """
        Determine what content should be summarized from state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Content to summarize or None
        """
        # Priority order for content to summarize:
        
        # 1. Explicit content in metadata
        explicit_content = state.execution_metadata.get("content_to_summarize")
        if explicit_content:
            return explicit_content

        # 2. Web search results
        search_results = state.execution_metadata.get("web_search_results")
        if search_results and isinstance(search_results, dict):
            results_content = []
            for result in search_results.get("results", []):
                if result.get("content"):
                    results_content.append(result["content"])
            if results_content:
                return "\n\n".join(results_content)

        # 3. Search results string
        if state.search_results:
            return state.search_results

        # 4. Conversation history
        if state.messages:
            conversation_parts = []
            for message in state.messages:
                if hasattr(message, 'content') and message.content:
                    role = getattr(message, 'type', 'unknown')
                    conversation_parts.append(f"{role}: {message.content}")
            if conversation_parts:
                return "\n\n".join(conversation_parts)

        return None

    async def summarize_search_results(self, state: WorkflowState) -> WorkflowState:
        """
        Specifically summarize web search results with synthesis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with synthesized search summary
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for search summarization")

            # Get search results and query
            search_results = state.execution_metadata.get("web_search_results", {}).get("results", [])
            query = self._extract_search_query_from_state(state)

            if not search_results:
                self.logger.info("No search results found for summarization")
                return state

            self.logger.info(
                "Summarizing search results",
                user_id=user_id,
                result_count=len(search_results),
                query=query[:100] if query else "unknown"
            )

            # Generate search results summary using the agent
            synthesis_result = await self.agent.summarize_search_results(
                search_results=search_results,
                user_id=user_id,
                query=query or "search query"
            )

            # Store comprehensive synthesis in state
            state.search_results = synthesis_result.get("summary", "")
            state.execution_metadata["search_synthesis"] = synthesis_result
            state.execution_metadata["search_summarization_completed"] = True

            self.logger.info(
                "Search results summarization completed",
                user_id=user_id,
                summary_length=len(synthesis_result.get("summary", "")),
                key_points_count=len(synthesis_result.get("key_points", []))
            )

            return state

        except Exception as e:
            self.logger.error(
                "Search results summarization failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Search summarization failed: {str(e)}")
            state.execution_metadata["search_summarization_completed"] = False
            return state

    def _extract_search_query_from_state(self, state: WorkflowState) -> Optional[str]:
        """Extract search query from various state locations."""
        
        # Check metadata first
        query = state.execution_metadata.get("search_query")
        if query:
            return query

        # Check latest user message
        for message in reversed(state.messages):
            if hasattr(message, 'type') and message.type == "human":
                if hasattr(message, 'content') and message.content:
                    return str(message.content)

        return None

    async def summarize_conversation(self, state: WorkflowState) -> WorkflowState:
        """
        Summarize conversation history with structured output.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with conversation summary
        """
        try:
            user_id = getattr(state, "user_id", None)
            if not user_id:
                raise NodeExecutionError("User ID required for conversation summarization")

            if not state.messages:
                self.logger.info("No conversation history found for summarization")
                return state

            # Convert messages to format expected by agent
            messages = []
            for message in state.messages:
                if hasattr(message, 'content') and message.content:
                    role = getattr(message, 'type', 'user')
                    messages.append({
                        "role": role,
                        "content": str(message.content)
                    })

            focus = state.execution_metadata.get("conversation_focus", "key_decisions")

            self.logger.info(
                "Summarizing conversation",
                user_id=user_id,
                message_count=len(messages),
                focus=focus
            )

            # Generate conversation summary using the agent
            conversation_summary = await self.agent.summarize_conversation(
                messages=messages,
                user_id=user_id,
                focus=focus
            )

            # Store conversation summary in state
            state.execution_metadata["conversation_summary"] = conversation_summary
            state.execution_metadata["conversation_summarization_completed"] = True

            self.logger.info(
                "Conversation summarization completed",
                user_id=user_id,
                summary_length=len(conversation_summary.get("summary", ""))
            )

            return state

        except Exception as e:
            self.logger.error(
                "Conversation summarization failed",
                user_id=getattr(state, "user_id", "unknown"),
                error=str(e)
            )
            # Don't raise - add error to state and continue workflow
            state.error_details.append(f"Conversation summarization failed: {str(e)}")
            state.execution_metadata["conversation_summarization_completed"] = False
            return state