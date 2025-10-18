"""
Intent analysis and classification agent.
Performs comprehensive intent analysis following the capability-driven architecture.
Maps user requests to RequiredCapabilities and assesses computational complexity.
"""

import json
from typing import List, TYPE_CHECKING, cast

from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel

from models import (
    ChatResponse,
    IntentAnalysis,
    MessageContent,
    MessageRole,
    ModelProfile,
    PipelinePriority,
    Message,
    NodeMetadata,
    MessageContentType,
    Tool,
)
from composer.core.errors import IntentAnalysisError
from composer.utils.conversion import (
    normalize_message_input,
    convert_messages_to_langchain,
)
from utils.message import extract_message_text
from utils.grammar_generator import parse_structured_output
from .base_agent import BaseAgent

if TYPE_CHECKING:
    from runner import PipelineFactory


class ClassifierAgent(BaseAgent[List[IntentAnalysis]]):
    """
    Grammar-constrained LLM intent analysis agent for workflow routing and tool selection.

    This agent performs comprehensive intent analysis using structured LLM output with
    grammar constraints to ensure guaranteed schema validation. It analyzes user messages
    to determine primary intent, complexity levels, required capabilities, and computational
    requirements that drive LangGraph workflow routing decisions.

    Key Features:
    - Grammar-constrained structured output using llamacpp grammars
    - Type-safe IntentAnalysis model generation with schema validation
    - Adaptive search depth determination based on complexity assessment
    - Graceful fallback handling when model profiles unavailable
    - Integration with shared model profile utilities for configuration management

    The agent executes early in LangGraph workflows to guide tool selection,
    workflow type selection, and retrieval depth configuration through structured
    intent classification that eliminates JSON parsing errors.
    """

    def __init__(
        self,
        pipeline_factory: "PipelineFactory",
        profile: ModelProfile,
        node_metadata: NodeMetadata,
    ):
        """
        Initialize the intent classification agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating analysis pipelines
            profile: Model profile for intent analysis operations
            node_metadata: Node execution metadata for tracking
        """
        super().__init__(pipeline_factory, profile, node_metadata, "ClassifierAgent")
        self.logger.info("Intent classifier initialized with analysis model profile")

    async def analyze(
        self,
        messages: List[Message],
        available_static_tools: List[Tool],
    ) -> List[IntentAnalysis]:
        """
        Execute grammar-constrained LLM analysis with structured output.

        intent_analysis.yaml schema to ensure guaranteed structure validation.
        Current implementation uses JSON schema validation as fallback.

        Args:
            pipeline: Model pipeline with analysis profile
            user_query: User message text for analysis

        Returns:
            str: LLM response (should be grammar-constrained JSON)
        """

        class _Intnts(BaseModel):
            intents: List[IntentAnalysis]

        intnt_schema = _Intnts.model_json_schema()
        user_query = messages[-1].content if messages else ""

        # Build available tools context
        available_tools_context = ""
        if available_static_tools:
            tool_descriptions = []
            for tool in available_static_tools[
                :10
            ]:  # Limit to first 10 tools for context
                tool_descriptions.append(f"- {tool.name}: {tool.description}")

            available_tools_context = f"""
Available Static Tools ({len(available_static_tools)} total):
{chr(10).join(tool_descriptions)}
{f"... and {len(available_static_tools) - 10} more tools" if len(available_static_tools) > 10 else ""}

Consider these available tools when assessing:
- requires_tools: Set to true if the request can be fulfilled using available tools
- requires_custom_tools: Set to true ONLY if available tools are insufficient and custom tool creation is needed
- tool_complexity_score: Lower scores if available tools can handle the request
"""

        analysis_prompt = f"""
You are an expert intent classification system. Analyze the user request and output ONLY JSON.

Valid enumerations ONLY:
workflow_type (choose one per intent): [ {" | ".join(intnt_schema['$defs']['WorkflowType']['enum'])} ]
complexity_level (choose one per intent): [ {" | ".join(intnt_schema['$defs']['ComplexityLevel']['enum'])} ]
computational_requirements (choose one per intent): [ {" | ".join(intnt_schema['$defs']['ComputationalRequirement']['enum'])} ]

required_capabilities (functionality needed - choose many, one, or none):
{", ".join(intnt_schema['$defs']['RequiredCapability']['enum'])}
required_capabilities can be empty if none apply. It is usually empty for simple queries.
DO NOT invent capabilities or requirements - only use those listed above.

{available_tools_context}

Tool Assessment Guidelines:
- requires_tools: Set to true if the request needs external tools/APIs to be fulfilled (web search, file operations, calculations, etc.)
- requires_custom_tools: Set to true if existing tools won't suffice and custom tool creation is needed
- tool_complexity_score: Rate 0.0-1.0 based on how complex the required tooling would be
  * 0.0-0.3: Basic tools (search, simple calculations)  
  * 0.4-0.6: Moderate tools (data processing, API calls)
  * 0.7-1.0: Complex tools (custom integrations, specialized processing)

Scoring Guidelines:
- domain_specificity: 0.0-1.0 (0.0=general, 1.0=highly domain-specific)
- reusability_potential: 0.0-1.0 (0.0=one-time use, 1.0=highly reusable)
- confidence: 0.0-1.0 (confidence in your analysis)

Instructions:
1. Decompose only if there are clearly separable sub-tasks; else one intent in the intents array.
2. Each element in intents must follow the enumerations exactly.
3. Omit response_format / technical_domain unless clearly implied.
4. All boolean fields (requires_tools, requires_custom_tools) must be explicitly set.
5. All required numeric fields must be provided as numbers (not strings).
6. Output strictly valid JSON. No prose, no markdown, no comments.

User Request: {user_query}

IMPORTANT: Return JSON that is valid against this schema:
{json.dumps(intnt_schema)}

If multiple intents are needed, include additional objects in the intents array.
"""
        msgs = []
        msgs.extend(messages)
        msgs.append(analysis_prompt)
        result = await self.run(
            messages=msgs,
            tools=None,
            priority=PipelinePriority.HIGH,
            grammar=_Intnts,
        )

        txt = extract_message_text(result.message) if result and result.message else ""
        if not txt.strip():
            raise IntentAnalysisError("Empty intent analysis response")

        intents = parse_structured_output(txt, _Intnts)
        return intents.intents

    async def generate_title(
        self,
        messages: List[Message],
    ) -> str:
        """
        Generate a concise, descriptive title for a conversation based on its messages.

        Args:
            messages: List of conversation messages to analyze
            circuit_breaker: Optional circuit breaker configuration

        Returns:
            str: Generated conversation title (2-6 words)

        Raises:
            IntentAnalysisError: When title generation fails
        """
        try:
            # Extract text from all messages for context
            conversation_text = ""
            for message in messages[-5:]:  # Use last 5 messages for context
                text = extract_message_text(message)
                if text.strip():
                    role = "User" if message.role.value == "user" else "Assistant"
                    conversation_text += f"{role}: {text}\n"

            if not conversation_text.strip():
                return "New Conversation"

            title_prompt = f"""
Generate a concise, descriptive title for this conversation. The title should:
- Be 2-6 words maximum
- Capture the main topic or purpose
- Be clear and professional
- Not include quotes or special characters
- Be suitable as a conversation label

Conversation:
{conversation_text}

Title:"""

            # Use pipeline with title generation
            with self.pipeline_factory.pipeline(
                self.profile,
                PipelinePriority.MEDIUM,
            ) as pipeline:

                system_prompt = getattr(self.profile, "system_prompt", "")
                for msg in messages:
                    if msg.role == MessageRole.SYSTEM:
                        system_prompt += f"\n\n{extract_message_text(msg)}"

                agent = create_agent(
                    model=cast(BaseChatModel, pipeline),
                    system_prompt=system_prompt,
                )

                normalized_messages = convert_messages_to_langchain(
                    normalize_message_input(title_prompt)
                )

                result = await agent.ainvoke({"messages": normalized_messages})  # type: ignore

                # Convert agent result to ChatResponse
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1]
                    response = ChatResponse(
                        message=Message(
                            content=[
                                MessageContent(
                                    text=(
                                        str(last_message.content)
                                        if hasattr(last_message, "content")
                                        else ""
                                    ),
                                    type=MessageContentType.TEXT,
                                )
                            ],
                            role=MessageRole.ASSISTANT,
                        ),
                        done=True,
                    )
                else:
                    response = ChatResponse(
                        message=Message(
                            content=[
                                MessageContent(
                                    text="Agent completed without output",
                                    type=MessageContentType.TEXT,
                                )
                            ],
                            role=MessageRole.ASSISTANT,
                        ),
                        done=True,
                    )

                response.channels = self._node_metadata.model_dump()
                return (
                    extract_message_text(response.message)
                    if response and response.message
                    else ""
                )

        except Exception as e:
            self.logger.error(
                "Title generation failed", error=str(e), context="title_generation"
            )
            # Provide fallback title instead of raising error
            return "Conversation"
