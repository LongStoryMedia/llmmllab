"""
RAG routing nodes for adaptive retrieval-augmented generation.
Implements shallow/deep RAG routing based on intent classification per Phase 2 requirements.
"""

import asyncio
from typing import List, Dict, Any, Optional

from models import LangChainMessage, ModelProfileType

from db import storage
from utils.model_profile_utils import get_model_profile_for_task

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class RAGRouter:
    """
    Routes RAG execution based on intent classification and complexity requirements.
    
    Implements conditional routing for SHALLOW vs DEEP RAG based on the
    rag_depth_config field in workflow state.
    """

    def __init__(self):
        """Initialize RAG router."""
        self.logger = composer_logger.bind(component="RAGRouter")

    def route_rag_depth(self, state: WorkflowState) -> str:
        """
        Determine RAG routing based on state configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Next node name: "execute_shallow_search" or "execute_deep_crawl_and_synthesize"
        """
        try:
            rag_depth = getattr(state, 'rag_depth_config', None)
            
            if not rag_depth:
                # Default to shallow search if no configuration
                return "execute_shallow_search"

            if rag_depth.upper() == "DEEP":
                self.logger.info(
                    "Routing to deep RAG",
                    user_id=getattr(state, 'user_id', 'unknown'),
                    reason="Complex query requires comprehensive search"
                )
                return "execute_deep_crawl_and_synthesize"
            else:
                self.logger.info(
                    "Routing to shallow RAG",
                    user_id=getattr(state, 'user_id', 'unknown'),
                    reason="Simple query using fast retrieval"
                )
                return "execute_shallow_search"

        except Exception as e:
            self.logger.error(
                "RAG routing failed, defaulting to shallow",
                error=str(e),
                user_id=getattr(state, 'user_id', 'unknown')
            )
            return "execute_shallow_search"


class ShallowRAGExecutor:
    """
    Executes shallow RAG using internal vector store retrieval.
    
    Fast, single-pass retrieval designed for simple queries and low latency.
    """

    def __init__(self, user_id: str):
        """
        Initialize shallow RAG executor.
        
        Args:
            user_id: User identifier for configuration and context
        """
        self.user_id = user_id
        self.logger = composer_logger.bind(component="ShallowRAGExecutor")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute shallow RAG retrieval.
        
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
            uc = await storage.get_service(storage.user_config).get_user_config(self.user_id)
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing shallow RAG search",
                user_id=self.user_id,
                query_length=len(query),
                max_results=search_config.max_sources
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
                
                # Add context message for the model
                context_message = LangChainMessage(
                    role="system",
                    content=f"Context from memory:\n\n{search_results}"
                )
                
                # Insert context before the latest user message
                if len(state.messages) > 0:
                    state.messages.insert(-1, context_message)
                else:
                    state.messages.append(context_message)

                self.logger.info(
                    "Shallow RAG completed",
                    user_id=self.user_id,
                    results_count=len(memories),
                    results_length=len(search_results)
                )
            else:
                state.search_results = "No relevant context found in memory."
                self.logger.info(
                    "No results found for shallow RAG",
                    user_id=self.user_id
                )

            return state

        except Exception as e:
            self.logger.error(
                "Shallow RAG execution failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Continue without context on error
            state.search_results = f"Memory search failed: {str(e)}"
            return state

    def _format_shallow_results(self, memories: List[Any]) -> str:
        """
        Format memory search results for shallow RAG.
        
        Args:
            memories: List of memory objects from search
            
        Returns:
            Formatted context string
        """
        if not memories:
            return "No relevant memories found."

        result_parts = []
        for i, memory in enumerate(memories[:3]):  # Limit to top 3 for shallow
            content = memory.content[:300]  # Truncate for brevity
            result_parts.append(f"{i+1}. {content}")

        return "\n\n".join(result_parts)


class DeepRAGExecutor:
    """
    Executes deep RAG with external search, crawling, and synthesis.
    
    Resource-intensive, multi-step retrieval for complex queries requiring
    comprehensive information gathering and analysis.
    """

    def __init__(self, user_id: str):
        """
        Initialize deep RAG executor.
        
        Args:
            user_id: User identifier for configuration and context
        """
        self.user_id = user_id
        self.logger = composer_logger.bind(component="DeepRAGExecutor")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute deep RAG with crawling and synthesis.
        
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
            uc = await storage.get_service(storage.user_config).get_user_config(self.user_id)
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing deep RAG search",
                user_id=self.user_id,
                query_length=len(query),
                max_sources=search_config.max_sources,
                full_content=search_config.retrieve_full_content
            )

            # Step 1: Memory search (internal context)
            memory_results = await self._search_memory_context(query)
            
            # Step 2: External web search (if enabled)
            web_results = await self._search_web_sources(query, search_config)
            
            # Step 3: Synthesize comprehensive results
            comprehensive_results = await self._synthesize_deep_results(
                query, memory_results, web_results, search_config
            )

            state.search_results = comprehensive_results

            # Add comprehensive context to conversation
            if comprehensive_results:
                context_message = LangChainMessage(
                    role="system",
                    content=f"Comprehensive research context:\n\n{comprehensive_results}"
                )
                
                # Insert before latest user message
                if len(state.messages) > 0:
                    state.messages.insert(-1, context_message)
                else:
                    state.messages.append(context_message)

            self.logger.info(
                "Deep RAG completed",
                user_id=self.user_id,
                results_length=len(comprehensive_results) if comprehensive_results else 0
            )

            return state

        except Exception as e:
            self.logger.error(
                "Deep RAG execution failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Fallback to memory-only search on error
            try:
                memory_results = await self._search_memory_context(query)
                state.search_results = memory_results or f"Deep search failed: {str(e)}"
            except:
                state.search_results = f"All search methods failed: {str(e)}"
            
            return state

    async def _search_memory_context(self, query: str) -> str:
        """
        Search internal memory for relevant context.
        
        Args:
            query: Search query
            
        Returns:
            Formatted memory context
        """
        try:
            memory_service = storage.get_service(storage.memory)
            
            memories = await memory_service.search_memories(
                user_id=self.user_id,
                query=query,
                limit=10,  # More results for deep search
                similarity_threshold=0.7
            )

            if memories:
                memory_parts = []
                for i, memory in enumerate(memories):
                    content = memory.content[:500]  # Longer excerpts for deep search
                    memory_parts.append(f"Memory {i+1}: {content}")
                
                return "INTERNAL MEMORY:\n" + "\n\n".join(memory_parts)
            else:
                return "No relevant internal memory found."

        except Exception as e:
            self.logger.error(
                "Memory search failed in deep RAG",
                user_id=self.user_id,
                error=str(e)
            )
            return f"Memory search error: {str(e)}"

    async def _search_web_sources(self, query: str, search_config: Any) -> str:
        """
        Search external web sources for additional context.
        
        Args:
            query: Search query
            search_config: User search configuration
            
        Returns:
            Formatted web search results
        """
        try:
            # Check if web search is enabled in user configuration
            if not getattr(search_config, 'enable_web_search', True):
                return "Web search disabled in user configuration."

            # Use existing web search tool from server
            # This would integrate with the web extraction service
            from server.services.web_extraction_service import WebExtractionService
            
            web_service = WebExtractionService()
            
            # Perform web search with configured parameters
            search_results = await web_service.search_and_extract(
                query=query,
                max_results=min(search_config.max_sources, 10),
                extract_full_content=search_config.retrieve_full_content
            )

            if search_results:
                web_parts = []
                for i, result in enumerate(search_results):
                    title = result.get('title', 'Unknown')
                    content = result.get('content', '')[:800]  # Substantial excerpts
                    url = result.get('url', '')
                    
                    web_parts.append(f"Source {i+1}: {title}\nURL: {url}\n{content}")
                
                return "WEB SOURCES:\n" + "\n\n".join(web_parts)
            else:
                return "No relevant web sources found."

        except Exception as e:
            self.logger.error(
                "Web search failed in deep RAG",
                user_id=self.user_id,
                error=str(e)
            )
            return f"Web search error: {str(e)}"

    async def _synthesize_deep_results(
        self, 
        query: str, 
        memory_results: str, 
        web_results: str, 
        search_config: Any
    ) -> str:
        """
        Synthesize comprehensive results from multiple sources.
        
        Args:
            query: Original search query
            memory_results: Internal memory search results
            web_results: External web search results
            search_config: User search configuration
            
        Returns:
            Synthesized comprehensive context
        """
        try:
            # Combine all sources
            all_sources = []
            
            if memory_results and "No relevant" not in memory_results:
                all_sources.append(memory_results)
            
            if web_results and "No relevant" not in web_results and "disabled" not in web_results:
                all_sources.append(web_results)

            if not all_sources:
                return "No relevant information found from any source."

            # Create comprehensive synthesis
            synthesis_parts = [
                f"COMPREHENSIVE SEARCH RESULTS FOR: {query}",
                "=" * 50,
                ""
            ]

            # Add each source section
            for source in all_sources:
                synthesis_parts.append(source)
                synthesis_parts.append("")

            # Add summary section
            synthesis_parts.extend([
                "SYNTHESIS:",
                f"Found information from {len(all_sources)} source type(s).",
                "Use this context to provide a comprehensive response."
            ])

            return "\n".join(synthesis_parts)

        except Exception as e:
            self.logger.error(
                "Result synthesis failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Return raw results if synthesis fails
            return f"{memory_results}\n\n{web_results}"