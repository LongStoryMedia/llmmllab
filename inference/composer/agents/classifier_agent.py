"""
Intent analysis and classification agent.
Performs comprehensive intent analysis following the capability-driven architecture.
Maps user requests to RequiredCapabilities and assesses computational complexity.
"""

from email import message
import json
from typing import List, TYPE_CHECKING, Optional

from pydantic import BaseModel

from models import (
    CircuitBreakerConfig,
    IntentAnalysis,
    ModelProfile,
    PipelinePriority,
    Message,
    NodeMetadata,
)
from composer.core.errors import IntentAnalysisError
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
        self, user_id: str, messages: List[Message]
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

        analysis_prompt = f"""
You are an expert intent classification system. Analyze the user request and output ONLY JSON.

Valid enumerations ONLY:
workflow_type (choose one per intent): [ {" | ".join(intnt_schema['$defs']['WorkflowType']['enum'])} ]
complexity_level (choose one per intent): [ {" | ".join(intnt_schema['$defs']['ComplexityLevel']['enum'])} ]

required_capabilities (functionality needed - choose many, one, or none):
{", ".join(intnt_schema['$defs']['RequiredCapability']['enum'])}
required_capabilities can be empty if none apply. It is usually empty for simple queries.
DO NOT invent capabilities or requirements - only use those listed above.

Instructions:
1. Decompose only if there are clearly separable sub-tasks; else one intent in the intents array.
2. Each element in intents must follow the enumerations exactly.
3. Omit response_format / technical_domain unless clearly implied.
4. Output strictly valid JSON. No prose, no markdown, no comments.

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
            user_id=user_id,
            tools=None,
            circuit_breaker=self.profile.circuit_breaker,
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
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
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
                self.profile, str, PipelinePriority.MEDIUM, circuit_breaker
            ) as pipeline:
                from runner import (  # pylint: disable=import-outside-toplevel
                    run_pipeline,
                )

                result = await run_pipeline(
                    messages=[title_prompt],
                    pipeline=pipeline,
                    tools=None,
                    grammar=None,
                )

                if not result or not result.message:
                    return "Untitled Conversation"

                title = extract_message_text(result.message).strip()

                # Clean up the title
                title = title.replace('"', "").replace("'", "").strip()

                # Ensure it's not too long
                words = title.split()
                if len(words) > 6:
                    title = " ".join(words[:6])

                # Fallback if empty
                if not title:
                    title = "New Conversation"

                self.logger.info(
                    "Title generated successfully",
                    title=title,
                    word_count=len(title.split()),
                    message_count=len(messages),
                )

                return title

        except Exception as e:
            self.logger.error(
                "Title generation failed", error=str(e), context="title_generation"
            )
            # Provide fallback title instead of raising error
            return "Conversation"
