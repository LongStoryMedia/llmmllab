from typing import Optional, List
from langchain_core.tools import tool

from models import DynamicTool, Tool
from db import DynamicToolStorage, storage
from composer.tools.dynamic.security import ToolSecurityValidator
from composer.tools.dynamic.generator import DynamicToolRunner
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ToolGeneratorTool")


@tool
async def tool_generator(task_description: str, user_id: str) -> str:
    """
    Generates a new dynamic tool to perform a specific task.
    Provide a detailed description of what the tool should do.
    """
    logger.info(
        "Starting dynamic tool generation process",
        task=task_description,
        user_id=user_id,
    )
    try:
        # Import the singleton registry manager
        from composer.tools.registry import registry_manager
        
        # We need to access the user registry but we need to provide the engineering_agent
        # This is a limitation of the current design - tool functions can't access context easily
        # For now, we'll need to pass None and handle this differently
        # TODO: Consider refactoring to use a more context-aware approach
        logger.error("tool_generator function needs refactoring for singleton pattern")
        return "Error: This tool needs to be updated to work with the singleton pattern. Please contact system administrator."

        tool_storage: Optional[DynamicToolStorage] = storage.dynamic_tool
        if not tool_storage:
            return "Error: DynamicToolStorage is not available."

        engineering_agent = tool_registry.engineering_agent
        if not engineering_agent:
            return "Error: EngineeringAgent is not available in the ToolRegistry."

        # 1. Get existing static tools to provide as context to the engineering agent
        static_tools: List[Tool] = await tool_registry.get_static_tool_instances(
            user_id=user_id
        )

        # 2. Use EngineeringAgent to generate the tool specification
        logger.info("Invoking EngineeringAgent to generate tool specification...")
        generated_specs: List[DynamicTool] = (
            await engineering_agent.generate_dynamic_tool_specification(
                user_query=task_description,
                user_id=user_id,
                static_tools=static_tools,
            )
        )

        if not generated_specs:
            return (
                "Error: The engineering agent failed to generate a tool specification."
            )

        new_tool_spec = generated_specs[0]
        logger.info(
            "EngineeringAgent returned tool specification",
            tool_name=new_tool_spec.name,
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
            return (
                f"Error: Generated code is not secure. Validation failed: {error_msg}"
            )
        logger.info("Security validation passed.")

        # 4. Store the new tool in the database
        logger.info("Storing new dynamic tool in the database...")
        new_tool_spec.user_id = user_id

        created_tool = await tool_storage.create_tool(new_tool_spec)
        if not created_tool:
            return "Error: Failed to store the new tool in the database."
        logger.info(
            "Successfully stored new tool",
            tool_name=created_tool.name,
            tool_id=str(created_tool.id),
        )

        # 5. Register the executable tool instance in the registry
        executable_tool = DynamicToolRunner(created_tool)
        await tool_registry.register_dynamic_tool_instance(
            tool_id=str(created_tool.id), tool_instance=created_tool
        )
        # Also add it to the executable tools for the current run
        tool_registry.executable_tools[executable_tool.name] = executable_tool

        logger.info(
            "New dynamic tool is now registered and available for execution.",
            tool_name=executable_tool.name,
        )

        return f"Successfully created and registered new tool: '{created_tool.name}' (ID: {created_tool.id})."

    except Exception as e:
        logger.error(
            "An unexpected error occurred during tool generation", exc_info=True
        )
        return f"An unexpected error occurred: {e}"
