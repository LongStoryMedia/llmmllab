"""
Summarization Node for LangGraph workflows.
Wraps SummarizationAgent to provide content summarization within workflows.
"""

from typing import Optional

from composer.agents.summarization_agent import SummarizationAgent
from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


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