"""
Agentic workflow implementation with dynamic tool generation
"""

import logging
import asyncio
from typing import Dict, Optional

from langchain_core.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools import DuckDuckGoSearchRun

from models.message_content import MessageContent
from models.message_content_type import MessageContentType
from models.message_role import MessageRole
from models.message import Message
from server.context.conversation import ConversationContext
from runner.pipelines.base_pipeline import BasePipeline
from runner.pipelines.factory import PipelineFactory

from .generator import DynamicToolGenerator
from .dynamic_tool import DynamicTool
from .errors import handle_error as error_handler

logger = logging.getLogger(__name__)


class AgenticWorkflow:
    """Main agentic workflow with dynamic tool generation"""

    def __init__(
        self, pipeline_factory: PipelineFactory, conversation_ctx: ConversationContext
    ):
        self.pipeline_factory: PipelineFactory = pipeline_factory
        self.conversation_ctx: ConversationContext = conversation_ctx
        self.tool_generator: DynamicToolGenerator
        self.primary_pipeline: BasePipeline
        self.static_tools = []
        self.dynamic_tools: Dict[str, DynamicTool] = {}

    async def initialize(self, model_id: str):
        """
        Initialize the workflow with the specified model

        Args:
            model_id: ID of the model to use
        """
        # Get the primary pipeline
        self.primary_pipeline, _ = self.pipeline_factory.get_pipeline(model_id)

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
            name = "retrieve_memories"
            description = "Retrieve relevant memories from the conversation history"

            def __init__(self):
                super().__init__(name=self.name, description=self.description)

            def _run(
                self,
                query: str,
                run_manager: Optional[CallbackManagerForToolRun] = None,
                **kwargs,
            ) -> str:
                # Use the conversation context to retrieve memories
                memories = asyncio.run(conversation_ctx.retrieve_memories(query))

                if not memories:
                    return "No relevant memories found"

                memory_texts = []
                for memory in memories:
                    if hasattr(memory, "page_content"):
                        memory_texts.append(memory.page_content)
                    elif hasattr(memory, "text"):
                        memory_texts.append(memory.text)

                return "\n\n".join(memory_texts[:5])  # Limit to top 5

        return MemoryTool()

    async def process_request(self, user_message: Message) -> str:
        """
        Process a user request with agentic workflow

        Args:
            user_message: The user's message

        Returns:
            str: The response from the agentic workflow
        """
        try:
            # First, determine if we need to generate new tools
            needs_tool = await self._analyze_tool_needs(user_message)

            if needs_tool:
                # Extract message content text for tool generation
                message_text = "\n".join(
                    [
                        c.text
                        for c in user_message.content
                        if hasattr(c, "text") and c.text
                    ]
                )

                # Generate a dynamic tool
                tool = await self.tool_generator.generate_tool(message_text)
                if tool:
                    self.dynamic_tools[tool.name] = tool
                    logger.info(f"Generated dynamic tool for request: {tool.name}")

            # Combine static and dynamic tools
            all_tools = self.static_tools + list(self.dynamic_tools.values())

            # Create agent prompt
            agent_prompt = self._create_agent_prompt()

            # Use our adapter to convert the pipeline to an LLM compatible object
            # We're using structured_chat_agent as it's more flexible and works well with different LLM implementations

            # Create the agent using the adapter
            agent = create_structured_chat_agent(
                llm=self.primary_pipeline, tools=all_tools, prompt=agent_prompt
            )

            # Create agent executor
            agent_executor = AgentExecutor(
                agent=agent,
                tools=all_tools,
                verbose=True,
                max_iterations=10,
                early_stopping_method="generate",
            )

            # Execute the agent
            result = await agent_executor.ainvoke(
                {
                    "input": user_message,
                    "chat_history": [],  # Could include conversation history
                }
            )

            return result["output"]

        except (ValueError, TypeError, AttributeError) as e:
            # Handle specific errors
            error_handler(
                e,
                message="Specific error in agentic workflow",
                context={"user_message": user_message},
                raise_error=False,
            )
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
        except Exception as e:
            # Handle unexpected errors
            error_handler(
                e,
                message="Unexpected error in agentic workflow",
                context={"user_message": user_message},
                raise_error=False,
            )
            return f"I apologize, but I encountered an error processing your request. Our team has been notified."

    async def _analyze_tool_needs(self, user_message: Message) -> bool:
        """
        Analyze if the request needs a new dynamic tool

        Args:
            user_message: The user's message

        Returns:
            bool: True if a new tool is needed, False otherwise
        """
        analysis_prompt = f"""
Analyze this user request and determine if it requires creating a custom tool/function:

User request: {"\n".join([c.text for c in user_message.content if c.text])}

Consider if the request:
1. Involves complex calculations or data processing
2. Requires specific algorithms or logic
3. Needs custom data transformation
4. Would benefit from a reusable function

Available static tools:
- Web search
- Memory retrieval
- Basic conversation

Respond with only "YES" if a custom tool would be helpful, "NO" if existing tools are sufficient.
"""

        try:
            response = self.primary_pipeline.get(
                [
                    Message(
                        role=MessageRole.USER,
                        content=[
                            MessageContent(
                                type=MessageContentType.TEXT,
                                text=analysis_prompt,
                                url=None,
                            )
                        ],
                        conversation_id=user_message.conversation_id,
                    )
                ],
            )
            return "YES" in response.upper()
        except (ValueError, TypeError, AttributeError) as e:
            # Handle specific errors
            error_handler(
                e,
                message="Specific error analyzing tool needs",
                context={"user_message": user_message},
                raise_error=False,
            )
            return False
        except Exception as e:
            # Handle unexpected errors
            error_handler(
                e,
                message="Unexpected error analyzing tool needs",
                context={"user_message": user_message},
                raise_error=False,
            )
            logger.warning(f"Defaulting to no tool generation due to error: {e}")
            return False

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
