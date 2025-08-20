"""
Agentic workflow implementation with dynamic tool generation
"""

import logging
import asyncio
from typing import AsyncIterable, Dict, Optional, List
from datetime import datetime

from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun

from models.memory import Memory
from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.message import Message
from models.chat_response import ChatResponse
from server.context.conversation import ConversationContext
from server.utils.chat.message import to_lc_message
from runner.pipelines.base_pipeline import BasePipeline
from runner.pipelines.factory import pipeline_factory

from .generator import DynamicToolGenerator
from .dynamic_tool import DynamicTool
from .errors import handle_error as error_handler

logger = logging.getLogger(__name__)


class AgenticWorkflow:
    """Main agentic workflow with dynamic tool generation"""

    def __init__(self, conversation_ctx: ConversationContext):
        self.conversation_ctx: ConversationContext = conversation_ctx
        self.tool_generator: DynamicToolGenerator
        # self.primary_pipeline: BasePipeline
        self.static_tools = []
        self.dynamic_tools: Dict[str, DynamicTool] = {}

    async def initialize(self, model_id: str):
        """
        Initialize the workflow with the specified model

        Args:
            model_id: ID of the model to use
        """

        # Get the primary pipeline
        self.primary_pipeline, _ = pipeline_factory.get_pipeline(model_id)

        # Initialize tool generator
        self.tool_generator = DynamicToolGenerator(self.primary_pipeline)

        # Setup static tools
        await self._setup_static_tools()

    async def _setup_static_tools(self):
        """Setup static tools that are always available"""
        # Add web search tool
        search_tool = DuckDuckGoSearchRun()
        self.static_tools.append(search_tool)

        # Add memory retrieval tool
        memory_tool = self._create_memory_tool()
        self.static_tools.append(memory_tool)

    def _create_memory_tool(self) -> BaseTool:
        """
        Create a tool for retrieving memories

        Returns:
            BaseTool: Memory retrieval tool
        """
        conversation_ctx = self.conversation_ctx

        class MemoryTool(BaseTool):
            """
            Tool for retrieving memories from the conversation history
            """

            name = "retrieve_memories"
            description = "Retrieve relevant memories from the conversation history"

            def __init__(self):
                super().__init__(name=self.name, description=self.description)

            def _run(
                self,
                embeddings: List[List[float]],
                run_manager: Optional[CallbackManagerForToolRun] = None,
                **kwargs,
            ) -> List[Memory]:
                # Use the conversation context to retrieve memories
                return asyncio.run(
                    conversation_ctx.memory_context.retrieve_memories(embeddings)
                )

        return MemoryTool()

    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """
        Create the agent prompt template

        Returns:
            ChatPromptTemplate: The agent prompt template
        """
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant with access to various tools, including dynamically generated ones.
            
Use the available tools to help answer questions and accomplish tasks. When you need to perform calculations, 
data processing, or other complex operations, you can use the custom tools that have been generated.

Always explain your reasoning and what tools you're using.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
