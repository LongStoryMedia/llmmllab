"""
Engineering Agent for generating technical and engineering responses.
Provides core business logic for technical analysis, code generation, and engineering guidance.
"""

import re
import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from models import (
    IntentAnalysis,
    ModelProfile,
    PipelinePriority,
    TechnicalDomain,
    ResponseFormat,
    NodeMetadata,
    DynamicTool,
    Tool,
)
from composer.core.errors import NodeExecutionError
from utils.message import extract_message_text
from .base_agent import BaseAgent


if TYPE_CHECKING:
    from runner import PipelineFactory
    from db import DynamicToolStorage


NON_TOOL_NAME = "__NON_TOOL__"


class EngineeringAgent(BaseAgent[str]):
    """
    Engineering Agent for generating technical responses with grammar-constrained output.

    Provides core business logic for technical analysis, code generation, system design,
    and engineering guidance using configured engineering models. Supports tool integration
    and grammar constraints for structured outputs.
    """

    def __init__(
        self,
        pipeline_factory: "PipelineFactory",
        profile: ModelProfile,
        node_metadata: NodeMetadata,
        tool_storage: "DynamicToolStorage",
    ):
        """
        Initialize engineering agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating engineering pipelines
            profile: Model profile for engineering tasks
            node_metadata: Node execution metadata for tracking
        """
        super().__init__(pipeline_factory, profile, node_metadata, "EngineeringAgent")
        self.tool_storage = tool_storage

    async def generate_technical_response(
        self,
        query: str,
        user_id: str,
        domain: TechnicalDomain = TechnicalDomain.GENERAL_ENGINEERING,
        response_format: ResponseFormat = ResponseFormat.DETAILED_ANALYSIS,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> str:
        """
        Generate technical engineering response using configured engineering model.

        Args:
            query: Technical query or problem statement
            user_id: User identifier for model profile retrieval
            domain: Technical domain specialization
            response_format: Desired response format and structure
            tools: Optional tools available to the agent for enhanced capabilities
            grammar: Optional grammar constraints for structured output

        Returns:
            Technical response content
        """
        try:
            self.logger.info(
                "Generating technical response",
                user_id=user_id,
                query_length=len(query),
                domain=domain,
                response_format=response_format,
                has_tools=bool(tools),
                has_grammar=bool(grammar),
            )

            # Create engineering prompt based on domain and format
            prompt = await self._create_engineering_prompt(
                query=query, domain=domain, response_format=response_format
            )

            # Use BaseAgent's run method with proper parameters
            result = await self.run(
                messages=[prompt],
                tools=tools,
                priority=PipelinePriority.NORMAL,
                grammar=grammar,
            )

            # Extract response text
            response_text = (
                extract_message_text(result.message)
                if result and result.message
                else ""
            )

            if not response_text.strip():
                raise NodeExecutionError("Empty technical response generated")

            self.logger.info(
                "Generated technical response",
                user_id=user_id,
                response_length=len(response_text),
            )

            return response_text

        except Exception as e:
            self.logger.error(
                "Technical response generation failed",
                user_id=user_id,
                error=str(e),
            )
            raise NodeExecutionError(
                f"Technical response generation failed: {e}"
            ) from e

    async def analyze_system_architecture(
        self,
        system_description: str,
        user_id: str,
        analysis_focus: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze system architecture and provide recommendations.

        Args:
            system_description: Description of the system to analyze
            user_id: User identifier
            analysis_focus: Specific areas to focus analysis on
            tools: Optional tools for enhanced analysis capabilities

        Returns:
            Structured analysis results
        """
        try:
            self.logger.info(
                "Analyzing system architecture",
                user_id=user_id,
                description_length=len(system_description),
                focus_areas=analysis_focus or [],
            )

            # Create architecture analysis prompt
            analysis_prompt = await self._create_architecture_analysis_prompt(
                system_description, analysis_focus or []
            )

            # Generate analysis using technical response method
            analysis = await self.generate_technical_response(
                query=analysis_prompt,
                user_id=user_id,
                domain=TechnicalDomain.SYSTEM_ARCHITECTURE,
                response_format=ResponseFormat.DETAILED_ANALYSIS,
                tools=tools,
            )

            # Structure the analysis results
            structured_analysis = {
                "analysis": analysis,
                "system_description": system_description,
                "focus_areas": analysis_focus or [],
                "analysis_length": len(analysis),
                "recommendations": await self._extract_recommendations(analysis),
                "potential_issues": await self._extract_potential_issues(analysis),
            }

            self.logger.info(
                "System architecture analysis completed",
                user_id=user_id,
                analysis_length=len(analysis),
            )

            return structured_analysis

        except Exception as e:
            self.logger.error(
                "System architecture analysis failed", user_id=user_id, error=str(e)
            )
            raise NodeExecutionError(f"System architecture analysis failed: {e}") from e

    async def generate_code_solution(
        self,
        problem_statement: str,
        user_id: str,
        programming_language: Optional[str] = None,
        constraints: Optional[List[str]] = None,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generate code solution for engineering problem.

        Args:
            problem_statement: Description of the problem to solve
            user_id: User identifier
            programming_language: Preferred programming language
            constraints: Optional constraints for the solution
            tools: Optional tools for enhanced code generation
            grammar: Optional grammar for structured code output

        Returns:
            Code solution with explanation and metadata
        """
        try:
            self.logger.info(
                "Generating code solution",
                user_id=user_id,
                problem_length=len(problem_statement),
                language=programming_language,
                has_constraints=bool(constraints),
            )

            # Create code generation prompt
            code_prompt = await self._create_code_generation_prompt(
                problem_statement, programming_language, constraints or []
            )

            # Generate code solution
            solution = await self.generate_technical_response(
                query=code_prompt,
                user_id=user_id,
                domain=TechnicalDomain.SOFTWARE_DEVELOPMENT,
                response_format=ResponseFormat.CODE_SOLUTION,
                tools=tools,
                grammar=grammar,
            )

            # Structure the code solution
            code_solution = {
                "solution": solution,
                "problem_statement": problem_statement,
                "programming_language": programming_language,
                "constraints": constraints or [],
                "solution_length": len(solution),
                "code_blocks": await self._extract_code_blocks(solution),
                "explanation": await self._extract_explanation(solution),
            }

            self.logger.info(
                "Code solution generated successfully",
                user_id=user_id,
                solution_length=len(solution),
            )

            return code_solution

        except Exception as e:
            self.logger.error(
                "Code solution generation failed", user_id=user_id, error=str(e)
            )
            raise NodeExecutionError(f"Code solution generation failed: {e}") from e

    async def _create_engineering_prompt(
        self, query: str, domain: TechnicalDomain, response_format: ResponseFormat
    ) -> str:
        """Create engineering prompt based on domain and format."""

        domain_contexts = {
            TechnicalDomain.SOFTWARE_DEVELOPMENT: "As a software engineering expert, focus on code quality, design patterns, and best practices.",
            TechnicalDomain.SYSTEM_ARCHITECTURE: "As a system architecture expert, focus on scalability, reliability, and system design principles.",
            TechnicalDomain.DATA_ENGINEERING: "As a data engineering expert, focus on data pipelines, processing efficiency, and data quality.",
            TechnicalDomain.DEVOPS_INFRASTRUCTURE: "As a DevOps expert, focus on deployment, automation, monitoring, and infrastructure as code.",
            TechnicalDomain.SECURITY_ENGINEERING: "As a security engineering expert, focus on threat modeling, secure design, and security best practices.",
            TechnicalDomain.MACHINE_LEARNING: "As a machine learning engineering expert, focus on model design, data preprocessing, and ML pipelines.",
            TechnicalDomain.GENERAL_ENGINEERING: "As a general engineering expert, provide comprehensive technical guidance.",
        }

        format_instructions = {
            ResponseFormat.DETAILED_ANALYSIS: "Provide a detailed technical analysis with thorough explanations and context.",
            ResponseFormat.CODE_SOLUTION: "Provide working code with clear comments and explanations.",
            ResponseFormat.STEP_BY_STEP_GUIDE: "Provide a clear step-by-step guide with actionable instructions.",
            ResponseFormat.BEST_PRACTICES: "Focus on best practices, patterns, and recommended approaches.",
            ResponseFormat.TROUBLESHOOTING: "Provide systematic troubleshooting steps and diagnostic approaches.",
        }

        domain_context = domain_contexts.get(
            domain, domain_contexts[TechnicalDomain.GENERAL_ENGINEERING]
        )
        format_instruction = format_instructions.get(
            response_format, format_instructions[ResponseFormat.DETAILED_ANALYSIS]
        )

        prompt = f"""{domain_context}

{format_instruction}

Technical Query:
{query}

Please provide a comprehensive technical response addressing the query above. Include relevant technical details, examples where appropriate, and practical guidance."""

        return prompt

    async def _create_architecture_analysis_prompt(
        self, system_description: str, focus_areas: List[str]
    ) -> str:
        """Create system architecture analysis prompt."""

        focus_text = ""
        if focus_areas:
            focus_text = f" Pay special attention to: {', '.join(focus_areas)}."

        prompt = f"""Please analyze the following system architecture and provide detailed technical insights.{focus_text}

System Description:
{system_description}

Please provide:
1. Architectural strengths and weaknesses
2. Scalability considerations
3. Security implications
4. Performance characteristics
5. Maintenance and operational concerns
6. Recommended improvements

Analysis:"""

        return prompt

    async def _create_code_generation_prompt(
        self, problem_statement: str, language: Optional[str], constraints: List[str]
    ) -> str:
        """Create code generation prompt."""

        language_text = f" in {language}" if language else ""
        constraints_text = ""
        if constraints:
            constraints_text = "\n\nConstraints:\n" + "\n".join(
                f"- {constraint}" for constraint in constraints
            )

        prompt = f"""Generate a complete code solution{language_text} for the following problem:

Problem Statement:
{problem_statement}{constraints_text}

Please provide:
1. Working code with clear comments
2. Explanation of the approach
3. Time and space complexity analysis (if applicable)
4. Usage examples
5. Potential optimizations or alternatives

Code Solution:"""

        return prompt

    async def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from analysis text."""
        # Simple extraction - look for recommendation patterns
        lines = analysis.split("\n")
        recommendations = []

        for line in lines:
            line = line.strip()
            if any(
                keyword in line.lower()
                for keyword in ["recommend", "suggest", "should", "consider"]
            ):
                if len(line) > 10 and len(line) < 200:
                    recommendations.append(line)

        return recommendations[:5]  # Limit to top 5

    async def _extract_potential_issues(self, analysis: str) -> List[str]:
        """Extract potential issues from analysis text."""
        lines = analysis.split("\n")
        issues = []

        for line in lines:
            line = line.strip()
            if any(
                keyword in line.lower()
                for keyword in ["issue", "problem", "concern", "weakness", "limitation"]
            ):
                if len(line) > 10 and len(line) < 200:
                    issues.append(line)

        return issues[:5]  # Limit to top 5

    async def _extract_code_blocks(self, solution: str) -> List[str]:
        """Extract code blocks from solution text."""
        # Simple extraction - look for code block patterns
        code_blocks = re.findall(r"```[\w]*\n(.*?)\n```", solution, re.DOTALL)
        return code_blocks

    async def _extract_explanation(self, solution: str) -> str:
        """Extract explanation text from solution (non-code parts)."""
        # Remove code blocks and return remaining text
        explanation = re.sub(r"```[\w]*\n.*?\n```", "", solution, flags=re.DOTALL)
        return explanation.strip()

    async def generate_dynamic_tool_specification(
        self,
        user_query: str,
        user_id: str,
        intents: List[IntentAnalysis],  # IntentAnalysis type
        static_tools: List[Tool],  # List[Tool] type
    ) -> List[DynamicTool]:  # Optional[DynamicTool] type
        """
        Generate dynamic tool specification based on user query and intent analysis.

        Args:
            user_query: The user's query/request
            user_id: User identifier
            intent: Intent analysis containing workflow type, complexity, etc.
            static_tools: List of available static tools

        Returns:
            DynamicTool specification if needed, None if existing tools are sufficient
        """
        dynamic_tools = []
        try:
            self.logger.info(
                "Generating dynamic tool specification",
                query_length=len(user_query),
                has_static_tools=bool(static_tools),
            )

            for intent in intents:
                self.logger.info(
                    "Processing intent for dynamic tool generation",
                    workflow_type=intent.workflow_type,
                    complexity_level=intent.complexity_level,
                )

                # Create prompt for dynamic tool generation
                prompt = await self._create_tool_generation_prompt(
                    user_query=user_query,
                    intent=intent,
                    static_tools=static_tools,
                )

                # Use BaseAgent's run method to get LLM response
                result = await self.run(
                    messages=[prompt],
                    priority=PipelinePriority.NORMAL,
                    grammar=DynamicTool,
                )

                # Extract response text
                response_text = (
                    extract_message_text(result.message)
                    if result and result.message
                    else ""
                )

                if not response_text.strip():
                    self.logger.warning("Empty response from dynamic tool generation")
                    continue

                # Parse response and check if we should skip tool creation
                try:
                    parsed_response = json.loads(response_text)
                    if parsed_response.get("name") == NON_TOOL_NAME:
                        self.logger.info(
                            "Skipping dynamic tool creation",
                            reason="Existing tools sufficient",
                        )
                        continue

                    dt = DynamicTool(**parsed_response)

                    # Ensure user_id is set for persistence
                    if not dt.user_id:
                        dt.user_id = user_id  # type: ignore

                    dynamic_tools.append(dt)

                    # Persist the dynamic tool
                    await self.tool_storage.create_tool(dt)

                    self.logger.info(
                        "Dynamic tool specification generated successfully",
                        tool_name=dt.name,
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    self.logger.error(f"Failed to parse dynamic tool response: {e}")
                    continue

            return dynamic_tools

        except Exception as e:
            self.logger.error(
                "Dynamic tool specification generation failed",
                error=str(e),
            )
            raise NodeExecutionError(
                f"Dynamic tool specification generation failed: {e}"
            ) from e

    async def _create_tool_generation_prompt(
        self,
        user_query: str,
        intent: IntentAnalysis,
        static_tools: List[Tool],
    ) -> str:
        """Create prompt for dynamic tool generation."""
        static_tool_names = [getattr(tool, "name", str(tool)) for tool in static_tools]
        non_tool = DynamicTool(
            name=NON_TOOL_NAME,
            user_id="",
            code="",
            function_name="",
            description="",
            args_schema={},
            return_direct=False,
            tags=[],
        )

        prompt = f"""As a Tool Engineering Specialist, analyze the user's request and determine if a dynamic tool is needed beyond the available static tools.

User Query: {user_query}
Primary Intent: {intent.workflow_type}
Complexity Level: {intent.complexity_level}
Required Capabilities: {[str(cap) for cap in intent.required_capabilities]}

Available Static Tools: {static_tool_names}

CRITICAL ANALYSIS: Before creating any tool, determine if the user's request can be fulfilled using existing tools:
- If web_search is available and the query needs current information, do NOT create a dynamic tool
- If existing tools can handle the request, respond with: {non_tool.model_dump_json()}
- Only create a dynamic tool if there's a genuine capability gap

If a dynamic tool is genuinely needed, create a tool specification that:
1. Addresses specific capability gaps not covered by static tools
2. Is tailored to the user's query and intent
3. Has clear input/output schema definitions
4. Uses real, functional implementation (no fake APIs)
5. Considers security and validation requirements

Tool Requirements:
- Must be composable and re-usable
- Should have clear, typed input/output schema
- Should be efficient and focused on single responsibility
- Must not duplicate existing static tool functionality
- Use default error handling (handle_tool_error: false, handle_validation_error: false)

IMPORTANT: Use these exact default values for error handling:
- handle_tool_error: false (not true)
- handle_validation_error: false (not true)
- return_direct: false

Generate either a skip response or a structured tool specification in JSON format matching this schema: {json.dumps(DynamicTool.model_json_schema())}. Focus on practical implementation that directly addresses the user's needs."""
        return prompt
