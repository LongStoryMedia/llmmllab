"""
A tool that can generate other tools.
"""
from typing import Type, Optional, List
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool
import uuid

from models import DynamicTool, Tool
from composer.agents.engineering_agent import EngineeringAgent
from db import DynamicToolStorage, storage
from composer.tools.registry import ToolRegistry
from composer.tools.dynamic.security import ToolSecurityValidator
from composer.tools.dynamic.generator import DynamicToolRunner
from composer.tools.dynamic.serializer import RunnableToolComposer
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolGeneratorTool")


class ToolGeneratorInput(BaseModel):
    """Input schema for the ToolGeneratorTool."""

    task_description: str = Field(
        description="A detailed natural language description of the task the new tool should perform."
    )
    user_id: str = Field(description="The ID of the user requesting the tool.")


class ToolGeneratorTool(BaseTool):
    """
    A static tool that generates, validates, and registers new dynamic tools.

    This tool orchestrates the creation of new tools by:
    1. Using an EngineeringAgent to generate a DynamicTool specification from a task description.
    2. Validating the generated code for security.
    3. Storing the validated tool in the database.
    4. Making the new tool available for execution.
    """

    name: str = "tool_generator"
    description: str = (
        "Generates a new dynamic tool to perform a specific task. "
        "Provide a detailed description of what the tool should do."
    )
    args_schema: Type[BaseModel] = ToolGeneratorInput

    engineering_agent: EngineeringAgent
    tool_storage: DynamicToolStorage
    tool_registry: ToolRegistry

    def __init__(
        self,
        engineering_agent: EngineeringAgent,
        tool_storage: DynamicToolStorage,
        tool_registry: ToolRegistry,
    ):
        super().__init__()
        self.engineering_agent = engineering_agent
        self.tool_storage = tool_storage
        self.tool_registry = tool_registry

    def _run(self, task_description: str, user_id: str) -> str:
        """Synchronous wrapper for the async run method."""
        # This is a common pattern for LangChain tools that need to be async internally.
        # The actual execution happens in the async method.
        raise NotImplementedError("ToolGeneratorTool must be run asynchronously.")

    async def _arun(self, task_description: str, user_id: str) -> str:
        """
        Asynchronously generates, validates, and registers a new dynamic tool.

        Args:
            task_description: A detailed description of the tool's purpose.
            user_id: The ID of the user requesting the tool.

        Returns:
            A message indicating success or failure.
        """
        logger.info(
            "Starting dynamic tool generation process",
            task=task_description,
            user_id=user_id,
        )
        try:
            # 1. Get existing static tools to provide as context to the engineering agent
            static_tools: List[Tool] = (
                await self.tool_registry.get_static_tool_instances(user_id=user_id)
            )

            # 2. Use EngineeringAgent to generate the tool specification
            logger.info("Invoking EngineeringAgent to generate tool specification...")
            generated_specs: List[
                DynamicTool
            ] = await self.engineering_agent.generate_dynamic_tool_specification(
                user_query=task_description,
                user_id=user_id,
                static_tools=static_tools,
            )

            if not generated_specs:
                return "Error: The engineering agent failed to generate a tool specification."

            # For simplicity, we'll process the first generated tool.
            # A more complex implementation could handle multiple specs.
            new_tool_spec = generated_specs[0]
            logger.info(
                "EngineeringAgent returned tool specification", tool_name=new_tool_spec.name
            )

            # 3. Validate the generated code
            logger.info("Validating generated tool code for security...")
            is_valid, error_msg = ToolSecurityValidator.validate_code(new_tool_spec.code)
            if not is_valid:
                logger.warning(
                    "Generated code failed security validation",
                    tool_name=new_tool_spec.name,
                    error=error_msg,
                )
                return f"Error: Generated code is not secure. Validation failed: {error_msg}"
            logger.info("Security validation passed.")

            # 4. Store the new tool in the database
            logger.info("Storing new dynamic tool in the database...")
            # Ensure a UUID is set if not already present
            if not new_tool_spec.id:
                new_tool_spec.id = uuid.uuid4()
            
            new_tool_spec.user_id = user_id

            created_tool = await self.tool_storage.create_tool(new_tool_spec)
            if not created_tool:
                return "Error: Failed to store the new tool in the database."
            logger.info(
                "Successfully stored new tool",
                tool_name=created_tool.name,
                tool_id=str(created_tool.id),
            )

            # 5. Register the executable tool instance in the registry
            executable_tool = DynamicToolRunner(created_tool)
            await self.tool_registry.register_dynamic_tool_instance(
                tool_id=str(created_tool.id), tool_instance=executable_tool
            )
            logger.info(
                "New dynamic tool is now registered and available for execution.",
                tool_name=executable_tool.name,
            )

            return f"Successfully created and registered new tool: '{created_tool.name}' (ID: {created_tool.id})."

        except Exception as e:
            logger.error("An unexpected error occurred during tool generation", exc_info=True)
            return f"An unexpected error occurred: {e}"

