"""
Native composer tool integration following decoupling principles.
Uses only thin interfaces between composer and other components.
"""

import logging
from typing import List, AsyncGenerator, Union

from langchain_core.tools import BaseTool

from models import UserConfig
from .native_rag_tools import ComposerWebSearchTool, ComposerMemoryTool, ComposerSummarizationTool

logger = logging.getLogger(__name__)


class StandardToolProvider:
    """Provides native composer RAG tools following decoupling principles."""

    @staticmethod
    def get_standard_tools(user_config: UserConfig, conversation_id: int) -> List[BaseTool]:
        """Get native composer RAG tools using only thin interfaces."""
        return [
            ComposerWebSearchTool(user_config=user_config, conversation_id=conversation_id),
            ComposerMemoryTool(user_config=user_config, conversation_id=conversation_id),
            ComposerSummarizationTool(user_config=user_config, conversation_id=conversation_id),
        ]


class ModernToolManager:
    """
    Native composer tool management following decoupling principles.
    Uses only thin interfaces to other components.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def get_tools(
        self, user_config: UserConfig, conversation_id: int
    ) -> AsyncGenerator[Union[str, List[BaseTool]], None]:
        """
        Get native composer tools using only thin interfaces.
        
        Provides standard RAG tools implemented within composer component.
        Dynamic tool generation is handled by DynamicToolManager in composer/tools/dynamic/
        """
        try:
            # Provide native composer RAG tools
            standard_tools = StandardToolProvider.get_standard_tools(user_config, conversation_id)
            
            self.logger.info(f"Providing {len(standard_tools)} native composer tools")
            yield standard_tools

        except Exception as e:
            self.logger.error(f"Error getting tools: {e}", exc_info=True)
            yield f"Tool loading error: {str(e)}"


# Global tool manager instance
tool_manager = ModernToolManager()


async def get_tools(
    user_config: UserConfig, conversation_id: int
) -> AsyncGenerator[Union[str, List[BaseTool]], None]:
    """
    Main entry point for getting native composer tools using thin interfaces.
    """
    async for result in tool_manager.get_tools(user_config, conversation_id):
        yield result