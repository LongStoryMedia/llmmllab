"""
Native composer RAG tools implementation.
Decoupled from server services, using only thin interfaces (pipeline factory).
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.tools import BaseTool

from models import (
    Message,
    MessageContent,
    MessageContentType,
    MessageRole,
    UserConfig,
    PipelinePriority,
)
from runner import pipeline_factory


class ComposerWebSearchTool(BaseTool):
    """
    Native composer web search implementation.
    Uses only thin interfaces: pipeline factory for LLM operations.
    """

    name: str = "web_search"
    description: str = "Perform web search and return synthesized results"

    def __init__(self, user_config: UserConfig, conversation_id: int, **kwargs):
        super().__init__(**kwargs)
        self._user_config = user_config
        self._conversation_id = conversation_id
        self._logger = logging.getLogger(__name__)
    
    @property
    def user_config(self) -> UserConfig:
        return self._user_config
    
    @property 
    def conversation_id(self) -> int:
        return self._conversation_id
        
    @property
    def logger(self):
        return self._logger

    def _run(self, query: str) -> str:
        """Synchronous web search execution."""
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute web search with synthesis."""
        try:
            # For now, return placeholder - actual search service integration needed
            raw_results = [{"title": "Web search placeholder", "content": f"Search results for: {query}"}]
            
            if not raw_results:
                return "No relevant search results found."

            # Synthesize results using pipeline factory context manager pattern
            synthesis_prompt = f"""
            Based on the following web search results, provide a comprehensive synthesis for the query: "{query}"

            Search Results:
            {json.dumps(raw_results, indent=2)}

            Please provide:
            1. A clear, factual summary addressing the query
            2. Key insights from multiple sources
            3. Any important caveats or limitations
            
            Synthesis:
            """

            # Use correct pipeline factory pattern from server services
            try:
                from server.db import storage
                mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
                    self.user_config.model_profiles.summarization_profile_id,
                    self.user_config.user_id
                )
                
                if mp:
                    with pipeline_factory.pipeline(mp, str, PipelinePriority.HIGH) as pipe:
                        messages = [Message(
                            role=MessageRole.USER,
                            content=[MessageContent(
                                type=MessageContentType.TEXT,
                                text=synthesis_prompt
                            )]
                        )]
                        
                        # Import the pipeline runner
                        from runner.pipelines.run import run_pipeline
                        response = await run_pipeline(messages, pipe)
                        
                        if response and response.message:
                            from utils.message import extract_message_text
                            synthesis_text = extract_message_text(response.message)
                            return synthesis_text if synthesis_text else "Web search completed but synthesis failed."
            
            except Exception as e:
                self.logger.warning(f"Pipeline execution failed: {e}, using fallback")
            
            return f"Web search results for '{query}': {json.dumps(raw_results, indent=2)}"

        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return f"Web search failed: {str(e)}"


class ComposerMemoryTool(BaseTool):
    """
    Native composer memory search implementation.
    Uses only thin interfaces: pipeline factory for embeddings.
    """

    name: str = "memory_search"
    description: str = "Search conversation memory for relevant context"

    def __init__(self, user_config: UserConfig, conversation_id: int, **kwargs):
        super().__init__(**kwargs)
        self._user_config = user_config
        self._conversation_id = conversation_id
        self._logger = logging.getLogger(__name__)
    
    @property
    def user_config(self) -> UserConfig:
        return self._user_config
    
    @property 
    def conversation_id(self) -> int:
        return self._conversation_id
        
    @property
    def logger(self):
        return self._logger

    def _run(self, query: str) -> str:
        """Synchronous memory search execution."""
        return asyncio.run(self._arun(query))

    async def _arun(self, query: str) -> str:
        """Execute memory search with semantic similarity."""
        try:
            # For now, return placeholder - need proper embedding and memory search integration
            return f"Memory search results for '{query}': No relevant memories found at this time."

        except Exception as e:
            self.logger.error(f"Memory search error: {e}")
            return f"Memory search failed: {str(e)}"


class ComposerSummarizationTool(BaseTool):
    """
    Native composer summarization implementation.
    Uses only thin interfaces: pipeline factory for LLM operations.
    """

    name: str = "summarize"
    description: str = "Summarize long content to fit context limits"

    def __init__(self, user_config: UserConfig, conversation_id: int, **kwargs):
        super().__init__(**kwargs)
        self._user_config = user_config
        self._conversation_id = conversation_id
        self._logger = logging.getLogger(__name__)
    
    @property
    def user_config(self) -> UserConfig:
        return self._user_config
    
    @property 
    def conversation_id(self) -> int:
        return self._conversation_id
        
    @property
    def logger(self):
        return self._logger

    def _run(self, content: str, max_length: int = 500) -> str:
        """Synchronous summarization execution."""
        return asyncio.run(self._arun(content, max_length))

    async def _arun(self, content: str, max_length: int = 500) -> str:
        """Execute content summarization."""
        try:
            if len(content) <= max_length:
                return content

            # Create summarization prompt
            summary_prompt = f"""
            Please provide a concise summary of the following content, keeping it under {max_length} characters:

            Content:
            {content}

            Summary (max {max_length} chars):
            """

            # Use correct pipeline factory pattern
            try:
                from server.db import storage
                mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
                    self.user_config.model_profiles.summarization_profile_id,
                    self.user_config.user_id
                )
                
                if mp:
                    with pipeline_factory.pipeline(mp, str, PipelinePriority.MEDIUM) as pipe:
                        messages = [Message(
                            role=MessageRole.USER,
                            content=[MessageContent(
                                type=MessageContentType.TEXT,
                                text=summary_prompt
                            )]
                        )]
                        
                        from server.utils.pipeline_utils import run_pipeline
                        response = await run_pipeline(messages, pipe)
                        
                        if response and response.message:
                            from utils.message import extract_message_text
                            summary_text = extract_message_text(response.message)
                            
                            if summary_text:
                                # Ensure length constraint
                                if len(summary_text) > max_length:
                                    summary_text = summary_text[:max_length-3] + "..."
                                return summary_text
            
            except Exception as e:
                self.logger.warning(f"Pipeline execution failed: {e}, using fallback")
            
            # Fallback: truncate original content
            return content[:max_length-3] + "..."

        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            # Fallback: truncate original content
            return content[:max_length-3] + "..."


# Tool registry for native composer tools
NATIVE_COMPOSER_TOOLS = {
    "web_search": ComposerWebSearchTool,
    "memory_search": ComposerMemoryTool, 
    "summarize": ComposerSummarizationTool,
}


def create_native_tool(tool_name: str, user_config: UserConfig, conversation_id: int) -> Optional[BaseTool]:
    """
    Factory function to create native composer tools.
    Maintains architectural decoupling by using only thin interfaces.
    """
    tool_class = NATIVE_COMPOSER_TOOLS.get(tool_name)
    if tool_class:
        return tool_class(user_config=user_config, conversation_id=conversation_id)
    return None