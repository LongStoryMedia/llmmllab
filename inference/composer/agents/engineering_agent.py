"""
Engineering Agent for generating technical and engineering responses.
Provides core business logic for technical analysis, code generation, and engineering guidance.
"""

from typing import List, Optional, Dict, Any

from models import (
    CircuitBreakerConfig,
    ModelProfile,
    PipelinePriority,
    TechnicalDomain,
    ResponseFormat,
    NodeMetadata,
)
from composer.core.errors import NodeExecutionError
from utils.message import extract_message_text
from .base_agent import BaseAgent

from runner import PipelineFactory


class EngineeringAgent(BaseAgent[str]):
    """
    Engineering Agent for generating technical responses with grammar-constrained output.

    Provides core business logic for technical analysis, code generation, system design,
    and engineering guidance using configured engineering models. Supports tool integration
    and grammar constraints for structured outputs.
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
        profile: ModelProfile,
        node_metadata: NodeMetadata,
    ):
        """
        Initialize engineering agent with required dependencies.

        Args:
            pipeline_factory: Factory for creating engineering pipelines
            profile: Model profile for engineering tasks
            node_metadata: Node execution metadata for tracking
        """
        super().__init__(pipeline_factory, profile, node_metadata, "EngineeringAgent")

    async def execute_pipeline(self, stream: bool = False, **kwargs) -> str:
        """
        Execute engineering pipeline with the provided parameters.

        This is the standard interface for pipeline execution required by BaseAgent.

        Args:
            stream: Whether to stream the response (not applicable for engineering responses)
            **kwargs: Pipeline execution parameters, expected to include:
                - query: Technical query to analyze
                - user_id: User identifier
                - domain: Optional TechnicalDomain
                - response_format: Optional ResponseFormat
                - tools: Optional tools list
                - grammar: Optional grammar constraints
                - circuit_breaker: Optional CircuitBreakerConfig

        Returns:
            str: The engineering response
        """
        query = kwargs.get("query", "")
        user_id = kwargs.get("user_id", "")
        domain = kwargs.get("domain", TechnicalDomain.GENERAL_ENGINEERING)
        response_format = kwargs.get(
            "response_format", ResponseFormat.DETAILED_ANALYSIS
        )
        tools = kwargs.get("tools")
        grammar = kwargs.get("grammar")
        circuit_breaker = kwargs.get("circuit_breaker")

        if not query:
            raise NodeExecutionError(
                "query parameter is required for engineering analysis"
            )

        return await self.generate_technical_response(
            query=query,
            user_id=user_id,
            domain=domain,
            response_format=response_format,
            tools=tools,
            grammar=grammar,
            circuit_breaker=circuit_breaker,
        )

    async def generate_technical_response(
        self,
        query: str,
        user_id: str,
        domain: TechnicalDomain = TechnicalDomain.GENERAL_ENGINEERING,
        response_format: ResponseFormat = ResponseFormat.DETAILED_ANALYSIS,
        tools: Optional[List[Any]] = None,
        grammar: Optional[Any] = None,
        circuit_breaker: Optional[CircuitBreakerConfig] = None,
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
        # Lazy imports to avoid circular dependency
        from runner import run_pipeline  # pylint: disable=import-outside-toplevel

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

            # Use standard pipeline factory context manager pattern with optional tools/grammar
            with self.pipeline_factory.pipeline(
                self.profile, str, PipelinePriority.NORMAL, circuit_breaker
            ) as pipeline:
                res = await run_pipeline(prompt, pipeline, tools=tools, grammar=grammar)
                response = (
                    extract_message_text(res.message) if res and res.message else ""
                )

                self.logger.info(
                    "Technical response generated successfully",
                    user_id=user_id,
                    response_length=len(response),
                    domain=domain,
                )

                return response

        except Exception as e:
            self.logger.error(
                "Technical response generation failed",
                user_id=user_id,
                error=str(e),
                domain=domain,
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
        import re

        code_blocks = re.findall(r"```[\w]*\n(.*?)\n```", solution, re.DOTALL)
        return code_blocks

    async def _extract_explanation(self, solution: str) -> str:
        """Extract explanation text from solution (non-code parts)."""
        # Remove code blocks and return remaining text
        import re

        explanation = re.sub(r"```[\w]*\n.*?\n```", "", solution, flags=re.DOTALL)
        return explanation.strip()
