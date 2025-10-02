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


class SearchDepthRouter:
    """
    Routes search execution based on query complexity and intent classification.
    
    Implements conditional routing for SHALLOW vs DEEP search strategies based on the
    search_depth_config field in workflow state. This is part of a modern agentic
    system that adapts search strategy to query complexity.
    """

    def __init__(self):
        """Initialize search depth router."""
        self.logger = composer_logger.logger

    def route_search_depth(self, state: WorkflowState) -> str:
        """
        Determine search routing based on state configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name: "execute_shallow_search" or "execute_deep_crawl_and_synthesize"
        """
        try:
            # Check for explicit search depth configuration
            search_depth = getattr(state, 'search_depth_config', None)
            if not search_depth:
                # Check legacy field name for backwards compatibility
                search_depth = getattr(state, 'rag_depth_config', None)
            
            if not search_depth:
                # Default to shallow search if no configuration
                return "execute_shallow_search"

            if str(search_depth).upper() == "DEEP":
                self.logger.info(
                    "Routing to deep search strategy",
                    extra={
                        "user_id": getattr(state, 'user_id', 'unknown'),
                        "reason": "Complex query requires comprehensive information gathering"
                    }
                )
                return "execute_deep_crawl_and_synthesize"
            else:
                self.logger.info(
                    "Routing to shallow search strategy",
                    extra={
                        "user_id": getattr(state, 'user_id', 'unknown'),
                        "reason": "Simple query using fast retrieval"
                    }
                )
                return "execute_shallow_search"

        except Exception as e:
            self.logger.error(
                "Search depth routing failed, defaulting to shallow",
                extra={
                    "error": str(e),
                    "user_id": getattr(state, 'user_id', 'unknown')
                }
            )
            return "execute_shallow_search"


class ShallowSearchExecutor:
    """
    Executes shallow search using internal knowledge sources.
    
    Fast, single-pass information retrieval designed for simple queries requiring
    low latency responses. Uses internal memory and cached knowledge sources.
    """

    def __init__(self, user_id: str):
        """
        Initialize shallow search executor.
        
        Args:
            user_id: User identifier for configuration and context
        """
        self.user_id = user_id
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

            query = user_messages[-1].content

            # Retrieve user configuration from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel
            uc = await storage.get_service(storage.user_config).get_user_config(self.user_id)
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing shallow search strategy",
                extra={
                    "user_id": self.user_id,
                    "query_length": len(query),
                    "max_results": search_config.max_sources
                }
            )

            # Perform memory search using existing storage service
            memory_service = storage.get_service(storage.memory)
            
            memories = await memory_service.search_memories(
                user_id=self.user_id,
                query=query,
                limit=min(search_config.max_sources, 5),  # Limit for shallow search
                similarity_threshold=search_config.similarity_threshold
            )

            # Format search results
            if memories:
                search_results = self._format_shallow_results(memories)
                state.search_results = search_results
                
                # Add context message for the agentic system
                context_message = LangChainMessage(
                    role="system",
                    content=f"Context from knowledge sources:\n\n{search_results}"
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
                        "results_length": len(search_results)
                    }
                )
            else:
                state.search_results = "No relevant context found in knowledge sources."
                self.logger.info(
                    "No results found for shallow search",
                    extra={"user_id": self.user_id}
                )

            return state

        except Exception as e:
            self.logger.error(
                "Shallow search execution failed",
                extra={
                    "user_id": self.user_id,
                    "error": str(e)
                }
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


class DeepSearchExecutor:
    """
    Executes comprehensive search with multiple information sources and synthesis.
    
    Resource-intensive, multi-step information gathering for complex queries requiring
    comprehensive analysis from internal knowledge, external sources, and synthesis.
    """

    def __init__(self, user_id: str):
        """
        Initialize deep search executor.
        
        Args:
            user_id: User identifier for configuration and context
        """
        self.user_id = user_id
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

            query = user_messages[-1].content

            # Retrieve user configuration from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel
            uc = await storage.get_service(storage.user_config).get_user_config(self.user_id)
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing comprehensive search strategy",
                extra={
                    "user_id": self.user_id,
                    "query_length": len(query),
                    "max_sources": search_config.max_sources,
                    "full_content": search_config.retrieve_full_content
                }
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
                    content=f"Comprehensive information synthesis:\n\n{comprehensive_results}"
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
                    "results_length": len(comprehensive_results) if comprehensive_results else 0
                }
            )

            return state

        except Exception as e:
            self.logger.error(
                "Comprehensive search execution failed",
                extra={
                    "user_id": self.user_id,
                    "error": str(e)
                }
            )
            
            # Fallback to internal knowledge search on error
            try:
                internal_results = await self._search_internal_knowledge(query)
                state.search_results = internal_results or f"Comprehensive search failed: {str(e)}"
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
            from db import storage  # pylint: disable=import-outside-toplevel
            memory_service = storage.get_service(storage.memory)
            
            memories = await memory_service.search_memories(
                user_id=self.user_id,
                query=query,
                limit=10,  # More results for comprehensive search
                similarity_threshold=0.7
            )

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
                extra={
                    "user_id": self.user_id,
                    "error": str(e)
                }
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
            if not getattr(search_config, 'enable_web_search', True):
                return "External search disabled in user configuration."

            # Use existing web search service for external information
            from server.services.web_extraction_service import WebExtractionService
            
            web_service = WebExtractionService()
            
            # Perform external search with configured parameters
            search_results = await web_service.search_and_extract(
                query=query,
                max_results=min(search_config.max_sources, 10),
                extract_full_content=search_config.retrieve_full_content
            )

            if search_results:
                external_parts = []
                for i, result in enumerate(search_results):
                    title = result.get('title', 'Unknown')
                    content = result.get('content', '')[:800]  # Substantial excerpts
                    url = result.get('url', '')
                    
                    external_parts.append(f"Source {i+1}: {title}\nURL: {url}\n{content}")
                
                return "EXTERNAL SOURCES:\n" + "\n\n".join(external_parts)
            else:
                return "No relevant external sources found."

        except Exception as e:
            self.logger.error(
                "External search failed in comprehensive search",
                extra={
                    "user_id": self.user_id,
                    "error": str(e)
                }
            )
            return f"External search error: {str(e)}"

    async def _synthesize_comprehensive_results(
        self, 
        query: str, 
        internal_results: str, 
        external_results: str, 
        search_config: Any
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
            
            if external_results and "No relevant" not in external_results and "disabled" not in external_results:
                all_sources.append(external_results)

            if not all_sources:
                return "No relevant information found from any source."

            # Create comprehensive information synthesis
            synthesis_parts = [
                f"COMPREHENSIVE INFORMATION SYNTHESIS FOR: {query}",
                "=" * 60,
                ""
            ]

            # Add each information source section
            for source in all_sources:
                synthesis_parts.append(source)
                synthesis_parts.append("")

            # Add synthesis summary
            synthesis_parts.extend([
                "INFORMATION SYNTHESIS:",
                f"Gathered information from {len(all_sources)} source type(s).",
                "Use this comprehensive context to provide an informed response."
            ])

            return "\n".join(synthesis_parts)

        except Exception as e:
            self.logger.error(
                "Information synthesis failed",
                extra={
                    "user_id": self.user_id,
                    "error": str(e)
                }
            )
            
            # Return raw results if synthesis fails
            return f"{internal_results}\n\n{external_results}"