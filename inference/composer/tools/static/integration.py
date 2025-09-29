"""
Simplified tool integration for composer workflows.
Provides unified interface for static and dynamic tools.
"""

import logging
from typing import List, AsyncGenerator, Union

from langchain_core.tools import BaseTool

from server.services.context import ConversationContext
from .rag_tools import WebSearchTool, MemoryRetrievalTool, SummarizationTool

logger = logging.getLogger(__name__)


class StandardToolProvider:
    """Provides standard RAG tools with proper typing."""

    @staticmethod
    def get_standard_tools(conversation_ctx: ConversationContext) -> List[BaseTool]:
        """Get standard RAG tools for the conversation context."""
        return [
            WebSearchTool(conversation_ctx=conversation_ctx),
            MemoryRetrievalTool(conversation_ctx=conversation_ctx),
            SummarizationTool(conversation_ctx=conversation_ctx),
        ]


class ModernToolManager:
    """
    Simplified tool management system focused on coordination.
    Dynamic tool logic has been moved to composer/tools/dynamic/manager.py
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def get_tools(
        self, conversation_ctx: ConversationContext
    ) -> AsyncGenerator[Union[str, List[BaseTool]], None]:
        """
        Get available tools for the conversation context.
        
        This is a simplified version that provides standard RAG tools.
        Dynamic tool generation is handled by DynamicToolManager in composer/tools/dynamic/
        """
        try:
            # Provide standard RAG tools
            standard_tools = StandardToolProvider.get_standard_tools(conversation_ctx)
            
            self.logger.info(f"Providing {len(standard_tools)} standard tools")
            yield standard_tools

        except Exception as e:
            self.logger.error(f"Error getting tools: {e}", exc_info=True)
            yield f"Tool loading error: {str(e)}"


# Global tool manager instance
tool_manager = ModernToolManager()


async def get_tools(
    conversation_ctx: ConversationContext,
) -> AsyncGenerator[Union[str, List[BaseTool]], None]:
    """
    Main entry point for getting tools - delegates to the simplified tool manager.
    """
    async for result in tool_manager.get_tools(conversation_ctx):
        yield result