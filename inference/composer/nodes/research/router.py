"""
Search routing nodes for adaptive information retrieval in agentic systems.
Implements shallow/deep search routing based on query complexity and intent classification.
"""

from typing import List, Dict, Any, Optional

from models import LangChainMessage, ModelProfileType

# Database imports moved to method level to avoid circular dependencies

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError
from composer.utils.extraction import extract_content_from_langchain_message


class ResearchRouter:
    """
    Routes research execution based on query complexity and intent classification.

    Implements conditional routing for SHALLOW vs DEEP research strategies based on the
    search_depth_config field in workflow state. This is part of a modern agentic
    system that adapts research strategy to query complexity.
    """

    def __init__(self):
        """Initialize search depth router."""
        self.logger = composer_logger.logger

    def route_research_depth(self, state: WorkflowState) -> str:
        """
        Determine research routing based on state configuration.

        Args:
            state: Current workflow state

        Returns:
            Next node name: "execute_quick_research" or "execute_comprehensive_research"
        """
        try:
            # Check for explicit search depth configuration
            search_depth = getattr(state, "search_depth_config", None)
            if not search_depth:
                # Check legacy field name for backwards compatibility
                search_depth = getattr(state, "rag_depth_config", None)

            if not search_depth:
                # Default to quick research if no configuration
                return "execute_quick_research"

            if str(search_depth).upper() == "DEEP":
                self.logger.info(
                    "Routing to deep search strategy",
                    extra={
                        "user_id": getattr(state, "user_id", "unknown"),
                        "reason": "Complex query requires comprehensive information gathering",
                    },
                )
                return "execute_comprehensive_research"
            else:
                self.logger.info(
                    "Routing to shallow search strategy",
                    extra={
                        "user_id": getattr(state, "user_id", "unknown"),
                        "reason": "Simple query using fast retrieval",
                    },
                )
                return "execute_quick_research"

        except Exception as e:
            self.logger.error(
                "Search depth routing failed, defaulting to shallow",
                extra={
                    "error": str(e),
                    "user_id": getattr(state, "user_id", "unknown"),
                },
            )
            return "execute_quick_research"


class QuickResearchExecutor:
    """
    Executes shallow/quick research strategy.

    This executor implements a lightweight research approach suitable for simple queries
    that can be answered with minimal context. It focuses on speed and efficiency
    over comprehensiveness.
    """

    def __init__(self, user_id: str, pipeline_factory=None):
        """
        Initialize quick research executor.

        Args:
            user_id: User identifier for configuration and context
            pipeline_factory: Factory for creating embedding pipelines (optional)
        """
        self.user_id = user_id
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute shallow search strategy.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with search results
        """
        try:
            if not state.messages:
                return state

            # Get user query from latest message
            user_messages = [msg for msg in state.messages if msg.role == "user"]
            if not user_messages:
                return state

            query = extract_content_from_langchain_message(user_messages[-1])

            # Retrieve user configuration from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel

            uc = await storage.get_service(storage.user_config).get_user_config(
                self.user_id
            )
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing quick research strategy",
                extra={
                    "user_id": self.user_id,
                    "query_length": len(query),
                    "max_results": search_config.max_sources,
                },
            )

            # Perform memory search using embedding and memory agents
            from composer.agents.embedding_agent import (
                EmbeddingAgent,
            )  # pylint: disable=import-outside-toplevel
            from composer.agents.memory_agent import (
                MemoryAgent,
            )  # pylint: disable=import-outside-toplevel

            try:
                # Get embeddings for the query
                if not self.pipeline_factory:
                    raise NodeExecutionError(
                        "Pipeline factory not available for embedding generation"
                    )
                embedding_agent = EmbeddingAgent(self.pipeline_factory)
                query_embeddings = await embedding_agent.generate_embeddings(
                    [query], self.user_id
                )

                # Search memories using embeddings
                memory_agent = MemoryAgent()
                memories = await memory_agent.search_memories(
                    embeddings=query_embeddings,
                    user_id=self.user_id,
                    limit=min(search_config.max_sources, 5),  # Limit for shallow search
                    min_similarity=search_config.similarity_threshold,
                )
            except Exception as e:
                self.logger.warning(f"Memory search failed: {e}")
                memories = []

            # Format search results
            if memories:
                search_results = self._format_shallow_results(memories)
                state.search_results = search_results

                # Add context message for the agentic system
                context_message = LangChainMessage(
                    role="system",
                    content=f"Context from knowledge sources:\n\n{search_results}",
                )

                # Insert context before the latest user message
                if len(state.messages) > 0:
                    state.messages.insert(-1, context_message)
                else:
                    state.messages.append(context_message)

                self.logger.info(
                    "Shallow search completed successfully",
                    extra={
                        "user_id": self.user_id,
                        "results_count": len(memories),
                        "results_length": len(search_results),
                    },
                )
            else:
                state.search_results = "No relevant context found in knowledge sources."
                self.logger.info(
                    "No results found for shallow search",
                    extra={"user_id": self.user_id},
                )

            return state

        except Exception as e:
            self.logger.error(
                "Shallow search execution failed",
                extra={"user_id": self.user_id, "error": str(e)},
            )

            # Continue without context on error
            state.search_results = f"Knowledge search failed: {str(e)}"
            return state

    def _format_shallow_results(self, memories: List[Any]) -> str:
        """
        Format knowledge search results for shallow search.

        Args:
            memories: List of memory objects from search

        Returns:
            Formatted context string
        """
        if not memories:
            return "No relevant knowledge found."

        result_parts = []
        for i, memory in enumerate(memories[:3]):  # Limit to top 3 for shallow
            content = memory.content[:300]  # Truncate for brevity
            result_parts.append(f"{i+1}. {content}")

        return "\n\n".join(result_parts)


class ComprehensiveResearchExecutor:
    """
    Executes comprehensive search with multiple information sources and synthesis.

    Resource-intensive, multi-step information gathering for complex queries requiring
    comprehensive analysis from internal knowledge, external sources, and synthesis.
    """

    def __init__(self, user_id: str, pipeline_factory=None):
        """
        Initialize deep search executor.

        Args:
            user_id: User identifier for configuration and context
            pipeline_factory: Factory for creating embedding pipelines (optional)
        """
        self.user_id = user_id
        self.pipeline_factory = pipeline_factory
        self.logger = composer_logger.logger

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute comprehensive search with multi-source synthesis.

        Args:
            state: Current workflow state

        Returns:
            Updated workflow state with comprehensive search results
        """
        try:
            if not state.messages:
                return state

            # Get user query from latest message
            user_messages = [msg for msg in state.messages if msg.role == "user"]
            if not user_messages:
                return state

            query = extract_content_from_langchain_message(user_messages[-1])

            # Retrieve user configuration from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel

            uc = await storage.get_service(storage.user_config).get_user_config(
                self.user_id
            )
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing comprehensive search strategy",
                extra={
                    "user_id": self.user_id,
                    "query_length": len(query),
                    "max_sources": search_config.max_sources,
                    "full_content": search_config.retrieve_full_content,
                },
            )

            # Step 1: Internal knowledge search
            internal_results = await self._search_internal_knowledge(query)

            # Step 2: External information search (if enabled)
            external_results = await self._search_external_sources(query, search_config)

            # Step 3: Synthesize comprehensive results
            comprehensive_results = await self._synthesize_comprehensive_results(
                query, internal_results, external_results, search_config
            )

            state.search_results = comprehensive_results

            # Add comprehensive context to conversation
            if comprehensive_results:
                context_message = LangChainMessage(
                    role="system",
                    content=f"Comprehensive information synthesis:\n\n{comprehensive_results}",
                )

                # Insert before latest user message
                if len(state.messages) > 0:
                    state.messages.insert(-1, context_message)
                else:
                    state.messages.append(context_message)

            self.logger.info(
                "Comprehensive search completed successfully",
                extra={
                    "user_id": self.user_id,
                    "results_length": (
                        len(comprehensive_results) if comprehensive_results else 0
                    ),
                },
            )

            return state

        except Exception as e:
            self.logger.error(
                "Comprehensive search execution failed",
                extra={"user_id": self.user_id, "error": str(e)},
            )

            # Fallback to internal knowledge search on error
            try:
                internal_results = await self._search_internal_knowledge(query)
                state.search_results = (
                    internal_results or f"Comprehensive search failed: {str(e)}"
                )
            except:
                state.search_results = f"All search methods failed: {str(e)}"

            return state

    async def _search_internal_knowledge(self, query: str) -> str:
        """
        Search internal knowledge sources for relevant context.

        Args:
            query: Search query

        Returns:
            Formatted internal knowledge context
        """
        try:
            # Use embedding and memory agents for proper search
            from composer.agents.embedding_agent import (
                EmbeddingAgent,
            )  # pylint: disable=import-outside-toplevel
            from composer.agents.memory_agent import (
                MemoryAgent,
            )  # pylint: disable=import-outside-toplevel

            try:
                # Get embeddings for the query
                if not self.pipeline_factory:
                    raise NodeExecutionError(
                        "Pipeline factory not available for embedding generation"
                    )
                embedding_agent = EmbeddingAgent(self.pipeline_factory)
                query_embeddings = await embedding_agent.generate_embeddings(
                    [query], self.user_id
                )

                # Search memories using embeddings
                memory_agent = MemoryAgent()
                memories = await memory_agent.search_memories(
                    embeddings=query_embeddings,
                    user_id=self.user_id,
                    limit=10,  # More results for comprehensive search
                    min_similarity=0.7,
                )
            except Exception as e:
                self.logger.warning(
                    f"Memory search failed in internal knowledge search: {e}"
                )
                memories = []

            if memories:
                knowledge_parts = []
                for i, memory in enumerate(memories):
                    content = memory.content[:500]  # Longer excerpts for deep search
                    knowledge_parts.append(f"Knowledge {i+1}: {content}")

                return "INTERNAL KNOWLEDGE:\n" + "\n\n".join(knowledge_parts)
            else:
                return "No relevant internal knowledge found."

        except Exception as e:
            self.logger.error(
                "Internal knowledge search failed in comprehensive search",
                extra={"user_id": self.user_id, "error": str(e)},
            )
            return f"Internal knowledge search error: {str(e)}"

    async def _search_external_sources(self, query: str, search_config: Any) -> str:
        """
        Search external information sources for additional context.

        Args:
            query: Search query
            search_config: User search configuration

        Returns:
            Formatted external search results
        """
        try:
            # Check if external search is enabled in user configuration
            if not getattr(search_config, "enable_web_search", True):
                return "External search disabled in user configuration."

            # Get WebSearchConfig from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel
            from models.web_search_config import (
                WebSearchConfig,
            )  # pylint: disable=import-outside-toplevel

            # Retrieve user-specific web search configuration
            web_search_config = None
            try:
                user_config = await storage.get_service(
                    storage.user_config
                ).get_user_config(self.user_id)
                if user_config and hasattr(user_config, "web_search"):
                    web_search_config = user_config.web_search
            except Exception as e:
                self.logger.warning(
                    f"Failed to retrieve WebSearchConfig for user {self.user_id}: {e}"
                )

            # Use default config if no user-specific config found
            if not web_search_config:
                web_search_config = WebSearchConfig(
                    max_results=10,
                    engines=["google", "bing", "duckduckgo"],
                    timeout=60.0,
                    max_urls_deep=5,
                )

            # Use modern web search orchestration for external information
            from composer.graph.search import (
                create_web_search_subgraph,
                WebSearchState,
            )  # pylint: disable=import-outside-toplevel

            # Create subgraph for comprehensive search
            search_subgraph = await create_web_search_subgraph()

            # Configure search state
            search_state = WebSearchState(
                user_id=self.user_id,
                query=query,
                search_config=web_search_config,
                max_results=min(
                    search_config.max_sources, web_search_config.max_results
                ),
                batch_size=3,  # Conservative batch size for research
            )

            # Execute search
            final_state = await search_subgraph.ainvoke(search_state)

            # Convert results to expected format
            search_results = {
                "results": final_state.get("synthesized_results", []),
                "summary": final_state.get("search_summary", "No results found"),
            }

            if search_results and search_results.get("results"):
                external_parts = []
                for i, result in enumerate(search_results["results"]):
                    title = result.get("title", "Unknown")
                    content = result.get("content", "")[:800]  # Substantial excerpts
                    url = result.get("url", "")

                    external_parts.append(
                        f"Source {i+1}: {title}\nURL: {url}\n{content}"
                    )

                # Include search summary
                summary = search_results.get("summary", "")
                results_text = "EXTERNAL SOURCES:\n" + "\n\n".join(external_parts)
                if summary:
                    results_text = f"{summary}\n\n{results_text}"

                return results_text
            else:
                return "No relevant external sources found."

        except Exception as e:
            self.logger.error(
                "External search failed in comprehensive search",
                extra={"user_id": self.user_id, "error": str(e)},
            )
            return f"External search error: {str(e)}"

    async def _synthesize_comprehensive_results(
        self,
        query: str,
        internal_results: str,
        external_results: str,
        search_config: Any,
    ) -> str:
        """
        Synthesize comprehensive results from multiple information sources.

        Args:
            query: Original search query
            internal_results: Internal knowledge search results
            external_results: External information search results
            search_config: User search configuration

        Returns:
            Synthesized comprehensive information context
        """
        try:
            # Combine all information sources
            all_sources = []

            if internal_results and "No relevant" not in internal_results:
                all_sources.append(internal_results)

            if (
                external_results
                and "No relevant" not in external_results
                and "disabled" not in external_results
            ):
                all_sources.append(external_results)

            if not all_sources:
                return "No relevant information found from any source."

            # Create comprehensive information synthesis
            synthesis_parts = [
                f"COMPREHENSIVE INFORMATION SYNTHESIS FOR: {query}",
                "=" * 60,
                "",
            ]

            # Add each information source section
            for source in all_sources:
                synthesis_parts.append(source)
                synthesis_parts.append("")

            # Add synthesis summary
            synthesis_parts.extend(
                [
                    "INFORMATION SYNTHESIS:",
                    f"Gathered information from {len(all_sources)} source type(s).",
                    "Use this comprehensive context to provide an informed response.",
                ]
            )

            return "\n".join(synthesis_parts)

        except Exception as e:
            self.logger.error(
                "Information synthesis failed",
                extra={"user_id": self.user_id, "error": str(e)},
            )

            # Return raw results if synthesis fails
            return f"{internal_results}\n\n{external_results}"
