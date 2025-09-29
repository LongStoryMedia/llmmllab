"""
LangChain RAG tools using interface layer functions directly.
No unnecessary abstractions - uses runner and composer interface functions.
"""

import asyncio
import json
from typing import List, Callable, Any, Awaitable

from langchain_core.tools import BaseTool
from pydantic import Field

from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


class WebSearchTool(BaseTool):
    """Web search tool using interface layer functions."""
    
    name: str = "web_search"
    description: str = "Perform web search using provided search function"
    
    search_function: Callable[[str], Awaitable[List[dict]]] = Field(..., exclude=True)
    
    def __init__(self, search_function: Callable[[str], Awaitable[List[dict]]], **kwargs):
        super().__init__(search_function=search_function, **kwargs)

    async def _arun(self, query: str, **kwargs) -> str:
        try:
            search_results = await self.search_function(query)
            
            if search_results:
                return f"Web search results: {json.dumps(search_results[:3], indent=2)}"
            else:
                return f"No web search results found for: {query}"
        except Exception as e:
            return f"Web search failed: {str(e)}"

    def _run(self, query: str, **kwargs) -> str:
        return asyncio.run(self._arun(query, **kwargs))


class MemoryRetrievalTool(BaseTool):
    """Memory retrieval tool using simple function."""
    
    name: str = "memory_retrieval" 
    description: str = "Retrieve memories using provided memory function"
    
    memory_function: Callable[[List[List[float]]], Awaitable[List[dict]]] = Field(..., exclude=True)
    
    def __init__(self, memory_function: Callable[[List[List[float]]], Awaitable[List[dict]]], **kwargs):
        super().__init__(memory_function=memory_function, **kwargs)

    async def _arun(self, embeddings: List[List[float]], **kwargs) -> str:
        try:
            if not embeddings:
                return "No embeddings provided for memory retrieval"
            
            memories = await self.memory_function(embeddings)
            
            if memories:
                return f"Retrieved memories: {json.dumps(memories, indent=2)}"
            else:
                return "No relevant memories found"
        except Exception as e:
            return f"Memory retrieval failed: {str(e)}"

    def _run(self, embeddings: List[List[float]], **kwargs) -> str:
        return asyncio.run(self._arun(embeddings, **kwargs))


class SummarizationTool(BaseTool):
    """Summarization tool using simple pipeline function."""
    
    name: str = "summarization"
    description: str = "Summarize content using provided pipeline function"
    
    pipeline_function: Callable[[str], Awaitable[str]] = Field(..., exclude=True)
    
    def __init__(self, pipeline_function: Callable[[str], Awaitable[str]], **kwargs):
        super().__init__(pipeline_function=pipeline_function, **kwargs)

    async def _arun(self, content: str, **kwargs) -> str:
        try:
            if not content:
                return "No content provided for summarization"
            
            summary_prompt = f"Please summarize: {content}"
            result = await self.pipeline_function(summary_prompt)
            
            if result:
                return f"Summary: {result}"
            
            return "Unable to generate summary"
        except Exception as e:
            return f"Summarization failed: {str(e)}"

    def _run(self, content: str, **kwargs) -> str:
        return asyncio.run(self._arun(content, **kwargs))


class RAGToolFactory:
    """Factory for creating RAG tools with interface layer functions."""
    
    def __init__(
        self,
        search_function: Callable[[str], Awaitable[List[dict]]],
        memory_function: Callable[[List[List[float]]], Awaitable[List[dict]]], 
        pipeline_function: Callable[[str], Awaitable[str]]
    ):
        self.search_function = search_function
        self.memory_function = memory_function
        self.pipeline_function = pipeline_function
    
    def create_web_search_tool(self) -> WebSearchTool:
        return WebSearchTool(search_function=self.search_function)
    
    def create_memory_tool(self) -> MemoryRetrievalTool:
        return MemoryRetrievalTool(memory_function=self.memory_function)
    
    def create_summarization_tool(self) -> SummarizationTool:
        return SummarizationTool(pipeline_function=self.pipeline_function)
    
    def create_all_tools(self) -> List[BaseTool]:
        return [
            self.create_web_search_tool(),
            self.create_memory_tool(), 
            self.create_summarization_tool()
        ]
