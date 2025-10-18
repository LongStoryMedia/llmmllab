from models import Tool, ModelProfileType, NodeMetadata
from composer.graph.state import WorkflowState
from composer.tools.registry import ToolRegistry
from composer.utils.extraction import extract_content_from_langchain_message
from composer.agents.engineering_agent import EngineeringAgent
from runner import PipelineFactory
from utils.model_profile import get_model_profile
from utils.logging import llmmllogger


class DynamicToolCreationNode:
    """
    Node responsible for creating dynamic tool specifications based on user queries and intent analysis.
    Uses the EngineeringAgent to generate tool specifications.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        pipeline_factory: PipelineFactory,
    ):
        self.tool_registry = tool_registry
        self.pipeline_factory = pipeline_factory
        self.logger = llmmllogger.logger.bind(component="DynamicToolCreationNode")

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """
        Create dynamic tool specifications based on user query and intent analysis.
        Uses the EngineeringAgent to generate tool specifications.
        """
        try:
            assert state.user_id
            assert state.intent_classification is not None
            assert state.available_tools is not None
            assert state.current_user_message is not None
            assert state.user_config

            self.logger.info(
                "Creating dynamic tool specification",
                user_id=state.user_id,
            )

            # Get engineering model profile
            mp = await get_model_profile(state.user_id, ModelProfileType.Engineering)

            # Create node metadata for the engineering agent
            node_metadata = NodeMetadata(
                node_name="DynamicToolCreationNode",
                node_id="dynamic_tool_creation",
                node_type="tool_creation",
                user_id=state.user_id,
                conversation_id=state.conversation_id,
            )

            # Create engineering agent
            engineering_agent = EngineeringAgent(
                pipeline_factory=self.pipeline_factory,
                profile=mp,
                node_metadata=node_metadata,
            )

            # Process each intent classification
            for intent in state.intent_classification:
                # Use engineering agent to generate dynamic tool specification
                dynamic_tool = await engineering_agent.generate_dynamic_tool_specification(
                    user_query=extract_content_from_langchain_message(
                        state.current_user_message
                    ),
                    user_id=state.user_id,
                    intent=intent,
                    static_tools=state.available_tools,
                )

                # If a dynamic tool was generated, convert it to a generic Tool for the state
                if dynamic_tool:
                    # Convert to generic Tool (agent only needs invocation metadata)
                    minimized_fields = {
                        "name": dynamic_tool.name,
                        "description": dynamic_tool.description,
                        "args_schema": dynamic_tool.args_schema,
                        "return_direct": dynamic_tool.return_direct,
                        "tags": dynamic_tool.tags,
                        "metadata": dynamic_tool.metadata,
                        "handle_tool_error": dynamic_tool.handle_tool_error,
                        "handle_validation_error": dynamic_tool.handle_validation_error,
                        "response_format": dynamic_tool.response_format,
                    }
                    state.dynamic_tools.append(Tool(**minimized_fields))

                    self.logger.info(
                        "Dynamic tool created and registered",
                        user_id=state.user_id,
                        tool_name=dynamic_tool.name,
                    )

        except Exception as e:
            self.logger.error(f"Dynamic tool creation failed: {e}")

        return state


