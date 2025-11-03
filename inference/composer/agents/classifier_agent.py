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
    convert_messages_to_base_langchain,
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
        
        # Extract text content from the last message using the utility function
        user_query = extract_message_text(messages[-1]) if messages else ""
        
        # DEBUG: Log the exact user query being analyzed
        self.logger.info(f"🔍 DEBUG_USER_QUERY: '{user_query}'")

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
technical_domain (set for ENGINEERING workflows): [ {" | ".join(intnt_schema['$defs']['TechnicalDomain']['enum'])} ]
response_format (set for ENGINEERING workflows): [ {" | ".join(intnt_schema['$defs']['ResponseFormat']['enum'])} ]

required_capabilities (functionality needed - choose many, one, or none):
{", ".join(intnt_schema['$defs']['RequiredCapability']['enum'])}
required_capabilities can be empty if none apply. It is usually empty for simple queries.
DO NOT invent capabilities or requirements - only use those listed above.

{available_tools_context}

Tool Assessment Guidelines:
- requires_tools: Set to true if the request needs external tools/APIs to be fulfilled (web search, file operations, calculations, etc.)
- requires_custom_tools: Set to true if existing tools won't suffice and custom tool creation is needed
- tool_complexity_score: Rate 0.0-1.0 based on how complex the required tooling would be (MUST be between 0.0 and 1.0 inclusive)
  * 0.0-0.3: Basic tools (search, simple calculations)  
  * 0.4-0.6: Moderate tools (data processing, API calls)
  * 0.7-1.0: Complex tools (custom integrations, specialized processing)

Scoring Guidelines (ALL scores MUST be between 0.0 and 1.0 inclusive):
- domain_specificity: 0.0-1.0 (0.0=general, 1.0=highly domain-specific)
- reusability_potential: 0.0-1.0 (0.0=one-time use, 1.0=highly reusable)  
- confidence: 0.0-1.0 (confidence in your analysis)

Workflow Classification Guidelines:
- ENGINEERING: Technical implementation, code solutions, architecture design, system design, API development, debugging, performance optimization, infrastructure setup, technical guidance, engineering best practices. Examples: "build a REST API", "design a microservices architecture", "implement a caching system", "optimize database performance", "create a CI/CD pipeline"
- RESEARCH: Pure information gathering about topics, literature reviews, fact-finding, market research, academic research. Examples: "what is machine learning", "research competitors", "find information about X", "summarize recent developments in Y"
- ANALYSIS: Data analysis, evaluation of existing systems, comparative analysis with specific data. Examples: "analyze this dataset", "evaluate system performance", "compare these options with metrics"
- CREATIVE: Content creation, writing, brainstorming, artistic tasks. Examples: "write a story", "create marketing copy", "brainstorm ideas"
- GENERAL: Simple questions, basic conversations, clarifications. Examples: "what time is it", "how are you", "explain briefly"

IMPORTANT: If a request asks for both technical information AND implementation/design guidance, classify as ENGINEERING, not RESEARCH.

Instructions:
1. Decompose only if there are clearly separable sub-tasks; else one intent in the intents array.
2. Each element in intents must follow the enumerations exactly.
3. For workflow_type=ENGINEERING, populate technical_domain and response_format when identifiable:
   - technical_domain: Choose the most appropriate domain (software_development for most code/API requests)
   - response_format: Choose based on what user wants (code_solution for implementation requests)
   - These fields help guide the engineering response but are optional
4. For other workflow types, omit response_format / technical_domain unless clearly implied.
5. All boolean fields (requires_tools, requires_custom_tools) must be explicitly set.
6. All required numeric fields must be provided as numbers (not strings).
7. Output strictly valid JSON. No prose, no markdown, no comments.
8. Technical requests asking for implementation guidance, code solutions, or system design should be ENGINEERING, not RESEARCH.

CRITICAL ENGINEERING FIELD REQUIREMENTS:
- REST API, FastAPI, API development → technical_domain: "software_development", response_format: "code_solution"
- System architecture, microservices → technical_domain: "system_architecture", response_format: "detailed_analysis"  
- Database optimization, queries → technical_domain: "data_engineering", response_format: "best_practices"
- Infrastructure, DevOps, CI/CD → technical_domain: "devops_infrastructure", response_format: "step_by_step_guide"

VALIDATION: Before outputting JSON, verify that:
- IF workflow_type is "engineering" THEN technical_domain MUST NOT be null/None
- IF workflow_type is "engineering" THEN response_format MUST NOT be null/None
- REJECT any engineering classification that leaves these fields empty

User Request: {user_query}

ENGINEERING FIELD GUIDANCE:
For engineering workflows, populate these fields when identifiable:
- technical_domain: Choose the most appropriate domain from the enum list
- response_format: Choose the most appropriate format from the enum list
These fields help guide the engineering agent but are optional.

IMPORTANT: Return JSON that is valid against this schema:
{json.dumps(intnt_schema)}

If multiple intents are needed, include additional objects in the intents array.
"""
        msgs = []
        msgs.extend(messages[:-1])  # All but last message
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

        # DEBUG: Log raw classifier output and parsed result
        self.logger.info(f"🔍 DEBUG_CLASSIFIER_RAW_OUTPUT: {txt[:500]}")
        
        intents = parse_structured_output(txt, _Intnts)
        
        # DEBUG: Log parsed intents with focus on engineering fields
        for i, intent in enumerate(intents.intents):
            self.logger.info(f"🔍 DEBUG_PARSED_INTENT_{i}: workflow_type={intent.workflow_type}, technical_domain={intent.technical_domain}, response_format={intent.response_format}")
            if intent.workflow_type == "engineering":
                self.logger.info(f"🔍 DEBUG_ENGINEERING_FIELDS: technical_domain={intent.technical_domain} (type: {type(intent.technical_domain)}), response_format={intent.response_format} (type: {type(intent.response_format)})")
        
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

                # Convert to native LangChain BaseMessage objects instead of our LangChainMessage

                normalized_messages = convert_messages_to_base_langchain(
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
                raw_title = (
                    extract_message_text(response.message)
                    if response and response.message
                    else ""
                )

                # Extract clean title from the response, handling structured output
                return self._extract_clean_title(raw_title)

        except Exception as e:
            self.logger.error(
                "Title generation failed", error=str(e), context="title_generation"
            )
            # Provide fallback title instead of raising error
            return "Conversation"

    def _extract_clean_title(self, raw_response: str) -> str:
        """
        Extract clean title from model response, handling structured output.

        Args:
            raw_response: Raw model response that may contain thinking tags, markdown, etc.

        Returns:
            str: Clean title string (2-6 words)
        """
        if not raw_response:
            return "Conversation"

        # Remove thinking tags and their content
        import re

        cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)

        # Remove markdown formatting
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)  # Bold
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)  # Italic
        cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)  # Code

        # Remove common prefixes
        cleaned = re.sub(
            r"^(Title:|Subject:|Topic:)\s*", "", cleaned, flags=re.IGNORECASE
        )

        # Clean up whitespace and line breaks
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Remove quotes
        cleaned = cleaned.strip("\"'")

        # Split into words and take reasonable length
        words = cleaned.split()
        if not words:
            return "Conversation"

        # Take first 6 words maximum for title
        title_words = words[:6]
        title = " ".join(title_words)

        # Fallback if empty or too short
        if not title or title.isspace():
            return "Conversation"

        return title
