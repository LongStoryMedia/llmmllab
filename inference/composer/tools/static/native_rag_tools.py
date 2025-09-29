"""
Native composer RAG tools with strict architectural decoupling.
All dependencies are injected via constructor parameters using Protocol interfaces.
"""

import asyncio
import json
from typing import List, Protocol, runtime_checkable

from langchain_core.tools import BaseTool
from pydantic import Field

from models.message import Message
from models.message_role import MessageRole
from models.message_content import MessageContent
from models.message_content_type import MessageContentType


@runtime_checkable
class PipelineInterface(Protocol):
    """Protocol for pipeline execution services."""
    
    async def execute_pipeline(self, message: Message, conversation_id: int) -> List[Message]:
        """Execute a pipeline and return generated messages."""
        ...


@runtime_checkable  
class SearchProviderInterface(Protocol):
    """Protocol for search providers."""
    
    async def search(self, message: Message, conversation_id: int) -> List[dict]:
        """Perform search and return search results."""
        ...


@runtime_checkable
class MemoryStoreInterface(Protocol):
    """Protocol for memory storage and retrieval."""
    
    async def retrieve_memories(self, embeddings: List[List[float]]) -> List[dict]:
        """Retrieve memories based on embeddings."""
        ...


class ComposerWebSearchTool(BaseTool):
    """Native composer web search tool using dependency injection."""
    
    name: str = "composer_web_search"
    description: str = "Perform web search using injected search provider"
    
    search_provider: SearchProviderInterface = Field(..., exclude=True)
    
    def __init__(self, search_provider: SearchProviderInterface, **kwargs):
        super().__init__(search_provider=search_provider, **kwargs)

    async def _arun(self, query: str, **kwargs) -> str:
        """Async implementation for web search."""
        try:
            message = Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=query)],
                conversation_id=kwargs.get("conversation_id", 0)
            )
            
            search_results = await self.search_provider.search(
                message, kwargs.get("conversation_id", 0)
            )
            
            if search_results:
                return f"Web search results: {json.dumps(search_results[:3], indent=2)}"
            else:
                return f"No web search results found for: {query}"
                
        except Exception as e:
            return f"Web search failed: {str(e)}"

    def _run(self, query: str, **kwargs) -> str:
        return asyncio.run(self._arun(query, **kwargs))


class ComposerMemoryTool(BaseTool):
    """Native composer memory retrieval tool using dependency injection."""
    
    name: str = "composer_memory_retrieval" 
    description: str = "Retrieve memories using injected memory store"
    
    memory_store: MemoryStoreInterface = Field(..., exclude=True)
    
    def __init__(self, memory_store: MemoryStoreInterface, **kwargs):
        super().__init__(memory_store=memory_store, **kwargs)

    async def _arun(self, embeddings: List[List[float]], **kwargs) -> str:
        """Async implementation for memory retrieval."""
        try:
            if not embeddings:
                return "No embeddings provided for memory retrieval"
            
            memories = await self.memory_store.retrieve_memories(embeddings)
            
            if memories:
                return f"Retrieved memories: {json.dumps(memories, indent=2)}"
            else:
                return "No relevant memories found"
                
        except Exception as e:
            return f"Memory retrieval failed: {str(e)}"

    def _run(self, embeddings: List[List[float]], **kwargs) -> str:
        return asyncio.run(self._arun(embeddings, **kwargs))


class ComposerSummarizationTool(BaseTool):
    """Native composer summarization tool using dependency injection."""
    
    name: str = "composer_summarization"
    description: str = "Summarize content using injected pipeline"
    
    pipeline: PipelineInterface = Field(..., exclude=True)
    
    def __init__(self, pipeline: PipelineInterface, **kwargs):
        super().__init__(pipeline=pipeline, **kwargs)

    async def _arun(self, content: str, **kwargs) -> str:
        """Async implementation for summarization."""
        try:
            if not content:
                return "No content provided for summarization"
            
            summary_prompt = f"Please summarize: {content}"
            message = Message(
                role=MessageRole.USER,
                content=[MessageContent(type=MessageContentType.TEXT, text=summary_prompt)],
                conversation_id=kwargs.get("conversation_id", 0)
            )
            
            result_messages = await self.pipeline.execute_pipeline(
                message, kwargs.get("conversation_id", 0)
            )
            
            if result_messages and len(result_messages) > 0:
                last_message = result_messages[-1]
                if last_message.content and len(last_message.content) > 0:
                    return f"Summary: {last_message.content[0].text}"
            
            return "Unable to generate summary"
                
        except Exception as e:
            return f"Summarization failed: {str(e)}"

    def _run(self, content: str, **kwargs) -> str:
        return asyncio.run(self._arun(content, **kwargs))


class ComposerToolFactory:
    """Factory for creating composer tools with proper dependency injection."""
    
    def __init__(
        self,
        search_provider: SearchProviderInterface,
        memory_store: MemoryStoreInterface, 
        pipeline: PipelineInterface
    ):
        self.search_provider = search_provider
        self.memory_store = memory_store
        self.pipeline = pipeline
    
    def create_web_search_tool(self) -> ComposerWebSearchTool:
        return ComposerWebSearchTool(search_provider=self.search_provider)
    
    def create_memory_tool(self) -> ComposerMemoryTool:
        return ComposerMemoryTool(memory_store=self.memory_store)
    
    def create_summarization_tool(self) -> ComposerSummarizationTool:
        return ComposerSummarizationTool(pipeline=self.pipeline)
    
    def create_all_tools(self) -> List[BaseTool]:
        return [
            self.create_web_search_tool(),
            self.create_memory_tool(), 
            self.create_summarization_tool()
        ]
