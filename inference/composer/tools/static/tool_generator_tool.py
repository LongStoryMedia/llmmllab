from typing import Optional, List
from langchain_core.tools import tool

from composer.models import DynamicTool, Tool
from composer.server import server
from composer.tools.dynamic.security import ToolSecurityValidator
from composer.tools.dynamic.generator import DynamicToolRunner
from composer.utils.logging import llmmllogger

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
        from composer.tools.registry import (  # pylint: disable=import-outside-toplevel
            registry_manager,
        )

        # Get the user-specific tool registry instance - it should already exist from graph building
        tool_registry = await registry_manager.get_existing_user_registry(user_id)
        if not tool_registry:
            logger.error(
                f"No cached registry found for user {user_id}. Registry must be initialized first."
            )
            return "Error: User registry not found. Please ensure the system is properly initialized for this user."

        tool_service: Optional[DynamicTool] = server.dynamic_tool
        if not tool_service:
            return "Error: DynamicTool service is not available."

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

        created_tool = await tool_service.create_tool(new_tool_spec)
        if not created_tool:
            return "Error: Failed to store the new tool in the database."
        logger.info(
            "Successfully stored new tool",
            tool_name=created_tool.name,
            tool_id=str(created_tool.id),
        )

        # 5. Register the executable tool instance in the registry
        executable_tool = DynamicToolRunner(created_tool)

        # Convert DynamicTool to Tool for registry storage
        tool_for_registry = Tool(
            name=created_tool.name,
            description=created_tool.description,
            args_schema=created_tool.args_schema,
            return_direct=created_tool.return_direct,
            tags=created_tool.tags,
            metadata=created_tool.metadata,
            handle_tool_error=created_tool.handle_tool_error,
            handle_validation_error=created_tool.handle_validation_error,
            response_format=created_tool.response_format,
        )

        await tool_registry.register_dynamic_tool_instance(
            tool_id=str(created_tool.id), tool_instance=tool_for_registry
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
