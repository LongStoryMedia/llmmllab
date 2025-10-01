"""
RAG execution nodes for retrieval-augmented generation.
Implements executor nodes that handle the actual RAG processing per Phase 2 requirements.
"""

import asyncio
from typing import List, Dict, Any, Optional

from models import LangChainMessage, ModelProfileType

from db import storage
from utils.model_profile_utils import get_model_profile_for_task

from composer.graph.state import WorkflowState
from composer.monitoring.logging import composer_logger
from composer.core.errors import NodeExecutionError


class RAGExecutorNode:
    """
    Base RAG executor node providing common functionality.
    
    Handles retrieval operations and result formatting for both shallow and deep RAG.
    """

    def __init__(self, user_id: str, execution_type: str = "standard"):
        """
        Initialize RAG executor node.
        
        Args:
            user_id: User identifier for configuration retrieval
            execution_type: Type of RAG execution (shallow, deep, standard)
        """
        self.user_id = user_id
        self.execution_type = execution_type
        self.logger = composer_logger.bind(
            component="RAGExecutorNode",
            execution_type=execution_type
        )

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Execute RAG retrieval based on configuration.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with retrieval results
        """
        try:
            if not state.messages:
                return state

            # Extract user query
            user_query = self._extract_user_query(state.messages)
            if not user_query:
                return state

            # Get user configuration from shared data layer
            uc = await storage.get_service(storage.user_config).get_user_config(self.user_id)
            search_config = uc.workflow_preferences.search_config

            self.logger.info(
                "Executing RAG retrieval",
                user_id=self.user_id,
                query_length=len(user_query),
                execution_type=self.execution_type
            )

            # Perform retrieval based on execution type
            search_results = await self._execute_retrieval(
                user_query, search_config, state
            )

            # Update state with results
            state.search_results = search_results

            # Add context to conversation if results found
            if search_results and "No relevant" not in search_results:
                await self._add_context_to_conversation(state, search_results)

            self.logger.info(
                "RAG execution completed",
                user_id=self.user_id,
                results_available=bool(search_results and "No relevant" not in search_results),
                results_length=len(search_results) if search_results else 0
            )

            return state

        except Exception as e:
            self.logger.error(
                "RAG execution failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Continue with error message in results
            state.search_results = f"RAG execution failed: {str(e)}"
            return state

    def _extract_user_query(self, messages: List[LangChainMessage]) -> str:
        """
        Extract the latest user query from messages.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Latest user query or empty string
        """
        user_messages = [msg for msg in messages if msg.role == "user"]
        return user_messages[-1].content if user_messages else ""

    async def _execute_retrieval(
        self, 
        query: str, 
        search_config: Any, 
        state: WorkflowState
    ) -> str:
        """
        Execute retrieval based on configuration and execution type.
        
        Args:
            query: User search query
            search_config: User search configuration
            state: Current workflow state
            
        Returns:
            Formatted search results
        """
        try:
            # Get memory service
            memory_service = storage.get_service(storage.memory)

            # Determine search parameters based on execution type
            if self.execution_type == "shallow":
                limit = min(search_config.max_sources, 3)
                threshold = search_config.similarity_threshold
                include_metadata = False
            elif self.execution_type == "deep":
                limit = min(search_config.max_sources, 15)
                threshold = max(search_config.similarity_threshold - 0.1, 0.5)  # Lower threshold for deep
                include_metadata = True
            else:
                limit = search_config.max_sources
                threshold = search_config.similarity_threshold
                include_metadata = True

            # Perform memory search
            memories = await memory_service.search_memories(
                user_id=self.user_id,
                query=query,
                limit=limit,
                similarity_threshold=threshold
            )

            # Format results based on execution type
            if memories:
                return await self._format_retrieval_results(
                    memories, include_metadata, query
                )
            else:
                return f"No relevant information found for: {query}"

        except Exception as e:
            self.logger.error(
                "Retrieval execution failed",
                user_id=self.user_id,
                error=str(e)
            )
            return f"Retrieval error: {str(e)}"

    async def _format_retrieval_results(
        self, 
        memories: List[Any], 
        include_metadata: bool, 
        query: str
    ) -> str:
        """
        Format retrieval results for context injection.
        
        Args:
            memories: Retrieved memory objects
            include_metadata: Whether to include metadata
            query: Original search query
            
        Returns:
            Formatted context string
        """
        if not memories:
            return f"No results found for: {query}"

        result_parts = [
            f"RETRIEVED CONTEXT FOR: {query}",
            "=" * 40,
            ""
        ]

        for i, memory in enumerate(memories):
            # Format each memory result
            content = memory.content
            
            # Truncate based on execution type
            if self.execution_type == "shallow":
                content = content[:300] + "..." if len(content) > 300 else content
            elif self.execution_type == "deep":
                content = content[:800] + "..." if len(content) > 800 else content

            result_entry = f"Result {i+1}:\n{content}"

            # Add metadata for deeper searches
            if include_metadata and hasattr(memory, 'created_at'):
                result_entry += f"\n(Retrieved: {memory.created_at})"

            result_parts.append(result_entry)
            result_parts.append("")  # Spacing

        # Add summary
        result_parts.extend([
            f"Retrieved {len(memories)} relevant result(s).",
            "Use this context to inform your response."
        ])

        return "\n".join(result_parts)

    async def _add_context_to_conversation(
        self, 
        state: WorkflowState, 
        search_results: str
    ):
        """
        Add retrieval context to the conversation.
        
        Args:
            state: Current workflow state
            search_results: Formatted search results
        """
        # Create context message
        context_message = LangChainMessage(
            role="system",
            content=search_results
        )

        # Insert context before the latest user message
        if len(state.messages) > 0:
            # Find the position of the last user message
            last_user_idx = None
            for i in range(len(state.messages) - 1, -1, -1):
                if state.messages[i].role == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                # Insert context before last user message
                state.messages.insert(last_user_idx, context_message)
            else:
                # No user message found, append to end
                state.messages.append(context_message)
        else:
            # Empty messages, just append
            state.messages.append(context_message)


class EnhancedRAGExecutor(RAGExecutorNode):
    """
    Enhanced RAG executor with external search capabilities.
    
    Extends base RAG functionality with web search, document processing,
    and advanced synthesis for comprehensive information retrieval.
    """

    def __init__(self, user_id: str):
        """
        Initialize enhanced RAG executor.
        
        Args:
            user_id: User identifier for configuration retrieval
        """
        super().__init__(user_id, execution_type="enhanced")

    async def _execute_retrieval(
        self, 
        query: str, 
        search_config: Any, 
        state: WorkflowState
    ) -> str:
        """
        Execute enhanced retrieval with multiple sources.
        
        Args:
            query: User search query
            search_config: User search configuration
            state: Current workflow state
            
        Returns:
            Comprehensive search results
        """
        try:
            # Step 1: Internal memory search
            memory_results = await self._search_internal_memory(
                query, search_config
            )

            # Step 2: External search if enabled
            external_results = await self._search_external_sources(
                query, search_config
            )

            # Step 3: Combine and synthesize results
            combined_results = await self._synthesize_results(
                query, memory_results, external_results
            )

            return combined_results

        except Exception as e:
            self.logger.error(
                "Enhanced retrieval failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Fallback to basic memory search
            return await super()._execute_retrieval(query, search_config, state)

    async def _search_internal_memory(
        self, 
        query: str, 
        search_config: Any
    ) -> Dict[str, Any]:
        """
        Search internal memory stores.
        
        Args:
            query: Search query
            search_config: User search configuration
            
        Returns:
            Memory search results
        """
        try:
            memory_service = storage.get_service(storage.memory)

            memories = await memory_service.search_memories(
                user_id=self.user_id,
                query=query,
                limit=search_config.max_sources,
                similarity_threshold=search_config.similarity_threshold
            )

            return {
                'source': 'internal_memory',
                'results': memories,
                'count': len(memories) if memories else 0
            }

        except Exception as e:
            self.logger.error(
                "Internal memory search failed",
                user_id=self.user_id,
                error=str(e)
            )
            return {
                'source': 'internal_memory',
                'results': [],
                'count': 0,
                'error': str(e)
            }

    async def _search_external_sources(
        self, 
        query: str, 
        search_config: Any
    ) -> Dict[str, Any]:
        """
        Search external web sources.
        
        Args:
            query: Search query
            search_config: User search configuration
            
        Returns:
            External search results
        """
        try:
            # Check if external search is enabled
            if not getattr(search_config, 'enable_web_search', True):
                return {
                    'source': 'external_web',
                    'results': [],
                    'count': 0,
                    'disabled': True
                }

            # Use web extraction service for external search
            from server.services.web_extraction_service import WebExtractionService
            
            web_service = WebExtractionService()
            
            search_results = await web_service.search_and_extract(
                query=query,
                max_results=min(search_config.max_sources, 5),
                extract_full_content=getattr(search_config, 'retrieve_full_content', False)
            )

            return {
                'source': 'external_web',
                'results': search_results or [],
                'count': len(search_results) if search_results else 0
            }

        except Exception as e:
            self.logger.error(
                "External search failed",
                user_id=self.user_id,
                error=str(e)
            )
            return {
                'source': 'external_web',
                'results': [],
                'count': 0,
                'error': str(e)
            }

    async def _synthesize_results(
        self, 
        query: str, 
        memory_results: Dict[str, Any], 
        external_results: Dict[str, Any]
    ) -> str:
        """
        Synthesize results from multiple sources.
        
        Args:
            query: Original search query
            memory_results: Internal memory search results
            external_results: External search results
            
        Returns:
            Synthesized comprehensive results
        """
        try:
            synthesis_parts = [
                f"COMPREHENSIVE SEARCH RESULTS FOR: {query}",
                "=" * 50,
                ""
            ]

            total_results = 0

            # Add internal memory results
            if memory_results['count'] > 0:
                synthesis_parts.extend([
                    "INTERNAL MEMORY:",
                    "-" * 20
                ])
                
                for i, memory in enumerate(memory_results['results'][:5]):
                    content = memory.content[:400]  # Truncate for readability
                    synthesis_parts.append(f"Memory {i+1}: {content}")
                    synthesis_parts.append("")
                
                total_results += memory_results['count']
            else:
                synthesis_parts.append("No relevant internal memory found.")
                synthesis_parts.append("")

            # Add external results if available
            if external_results['count'] > 0:
                synthesis_parts.extend([
                    "EXTERNAL SOURCES:",
                    "-" * 20
                ])
                
                for i, result in enumerate(external_results['results'][:3]):
                    title = result.get('title', 'Unknown Source')
                    content = result.get('content', '')[:400]
                    url = result.get('url', '')
                    
                    synthesis_parts.append(f"Source {i+1}: {title}")
                    if url:
                        synthesis_parts.append(f"URL: {url}")
                    synthesis_parts.append(content)
                    synthesis_parts.append("")
                
                total_results += external_results['count']
            elif external_results.get('disabled'):
                synthesis_parts.append("External search disabled in configuration.")
                synthesis_parts.append("")
            else:
                synthesis_parts.append("No relevant external sources found.")
                synthesis_parts.append("")

            # Add summary
            synthesis_parts.extend([
                "SUMMARY:",
                f"Found {total_results} total result(s) from {2 if memory_results['count'] > 0 and external_results['count'] > 0 else 1} source type(s).",
                "Use this comprehensive context to provide an informed response."
            ])

            return "\n".join(synthesis_parts)

        except Exception as e:
            self.logger.error(
                "Result synthesis failed",
                user_id=self.user_id,
                error=str(e)
            )
            
            # Return basic combination on synthesis failure
            parts = []
            if memory_results['count'] > 0:
                parts.append("Found relevant internal memory.")
            if external_results['count'] > 0:
                parts.append("Found relevant external sources.")
            
            return f"Search completed for: {query}\n" + " ".join(parts) if parts else f"No results found for: {query}"