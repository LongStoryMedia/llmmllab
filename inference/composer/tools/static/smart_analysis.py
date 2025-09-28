"""
Intelligent tool analysis system to reduce false positives in dynamic tool generation.
"""

import logging
import re
from typing import List, Set, Tuple

from models import IntentAnalysis, ComplexityLevel, RequiredCapability


class SmartIntentAnalyzer:
    """Advanced intent analyzer to reduce tool generation false positives."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Define patterns for different complexity levels
        self.trivial_patterns = [
            # Basic greetings and simple questions
            r"^(hi|hello|hey|thanks|thank you|bye|goodbye)\b",
            r"^(what is|who is|when is|where is)\s+\w+\s*\?*$",
            r"^(yes|no|ok|okay|sure|maybe)\s*\.?$",
            # Simple math (single operation, small numbers)
            r"^what.{0,10}is\s+\d{1,3}\s*[+\-*/]\s*\d{1,3}\s*\?*$",
            r"^\d{1,3}\s*[+\-*/]\s*\d{1,3}\s*=?\s*\?*$",
            # Basic information requests
            r"^(tell me about|what do you know about|explain)\s+\w+\s*$",
        ]

        self.simple_patterns = [
            # Simple calculations with clear operations
            r"\b(add|subtract|multiply|divide|sum)\s+\d+\s+(and|with|by)\s+\d+",
            r"\bwhat.{0,20}is\s+\d+\s*%\s+of\s+\d+",
            r"\bconvert\s+\d+\s+\w+\s+to\s+\w+",
            # Basic text operations
            r"\b(count|find|replace|remove)\s+(words|characters|letters)",
            r"\bmake.{0,10}(uppercase|lowercase|title case)",
        ]

        self.moderate_patterns = [
            # Multi-step calculations
            r"\bcalculate.{0,30}(compound|interest|percentage|growth|depreciation)",
            r"\bfind.{0,20}(average|mean|median|standard deviation)",
            r"\b(analyze|process|transform).{0,30}(data|list|numbers)",
            # Structured processing
            r"\b(parse|extract|format).{0,30}(json|xml|csv|table)",
            r"\b(sort|filter|group).{0,30}(by|according to|based on)",
        ]

        self.complex_patterns = [
            # Advanced algorithms
            r"\b(optimize|minimize|maximize|algorithm)",
            r"\b(machine learning|neural network|regression|classification)",
            r"\b(statistical|probability|distribution|correlation)",
            # Complex data processing
            r"\b(pattern|trend|anomaly|outlier)\s+(detection|analysis|identification)",
            r"\b(time series|forecasting|prediction|modeling)",
        ]

        self.specialized_patterns = [
            # Domain-specific requirements
            r"\b(financial|medical|legal|scientific|engineering)\s+(calculation|analysis|processing)",
            r"\b(regulatory|compliance|standard|protocol)\s+(check|validation|verification)",
            r"\b(custom|proprietary|specialized)\s+(algorithm|method|process)",
        ]

        # Available system capabilities
        self.system_capabilities = {
            RequiredCapability.BASIC_MATH: {
                "operations": [
                    "add",
                    "subtract",
                    "multiply",
                    "divide",
                    "percentage",
                    "basic_conversion",
                ],
                "limitations": ["complex_formulas", "advanced_functions"],
            },
            RequiredCapability.TEXT_PROCESSING: {
                "operations": ["search", "replace", "format", "basic_parsing"],
                "limitations": [
                    "complex_nlp",
                    "semantic_analysis",
                    "language_generation",
                ],
            },
            RequiredCapability.INFORMATION_RETRIEVAL: {
                "operations": ["lookup", "search", "retrieval"],
                "limitations": ["real_time_data", "specialized_databases"],
            },
            RequiredCapability.CONVERSATION_MEMORY: {
                "operations": ["recall", "context_retrieval", "history_search"],
                "limitations": ["long_term_memory", "complex_associations"],
            },
            RequiredCapability.WEB_SEARCH: {
                "operations": ["web_search", "information_gathering"],
                "limitations": ["deep_research", "specialized_sources"],
            },
            RequiredCapability.SUMMARIZATION: {
                "operations": ["text_summarization", "content_condensation"],
                "limitations": ["complex_analysis", "synthesis"],
            },
            RequiredCapability.REASONING: {
                "operations": ["logical_reasoning", "inference", "deduction"],
                "limitations": [
                    "complex_logic",
                    "formal_proofs",
                    "specialized_reasoning",
                ],
            },
            RequiredCapability.GENERAL_KNOWLEDGE: {
                "operations": ["factual_questions", "explanations", "general_advice"],
                "limitations": [
                    "specialized_knowledge",
                    "real_time_facts",
                    "personal_data",
                ],
            },
        }

    def analyze_intent(self, user_message: str) -> IntentAnalysis:
        """Perform comprehensive intent analysis."""
        # Determine complexity level
        complexity = self._assess_complexity(user_message)

        # Identify required capabilities
        required_capabilities = self._identify_required_capabilities(user_message)

        # Extract computational requirements
        computational_requirements = self._extract_computational_requirements(
            user_message
        )

        # Assess domain specificity
        domain_specificity = self._assess_domain_specificity(user_message)

        # Assess reusability potential
        reusability_potential = self._assess_reusability(user_message)

        # Determine primary intent
        primary_intent = self._extract_primary_intent(user_message)

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            user_message, complexity, required_capabilities, computational_requirements
        )

        return IntentAnalysis(
            primary_intent=primary_intent,
            complexity_level=complexity,
            required_capabilities=list(required_capabilities),  # Convert set to list
            computational_requirements=computational_requirements,
            domain_specificity=domain_specificity,
            reusability_potential=reusability_potential,
            confidence=confidence,
        )

    def _assess_complexity(self, message: str) -> ComplexityLevel:
        """Assess the computational complexity of the request."""
        message_lower = message.lower()

        # Check specialized patterns first (highest complexity)
        if any(
            re.search(pattern, message_lower) for pattern in self.specialized_patterns
        ):
            return ComplexityLevel.SPECIALIZED

        # Check complex patterns
        if any(re.search(pattern, message_lower) for pattern in self.complex_patterns):
            return ComplexityLevel.COMPLEX

        # Check moderate patterns
        if any(re.search(pattern, message_lower) for pattern in self.moderate_patterns):
            return ComplexityLevel.MODERATE

        # Check simple patterns
        if any(re.search(pattern, message_lower) for pattern in self.simple_patterns):
            return ComplexityLevel.SIMPLE

        # Check trivial patterns
        if any(re.search(pattern, message_lower) for pattern in self.trivial_patterns):
            return ComplexityLevel.TRIVIAL

        # Default assessment based on length and keywords
        if len(message.split()) <= 5:
            return ComplexityLevel.TRIVIAL
        elif len(message.split()) <= 15:
            return ComplexityLevel.SIMPLE
        else:
            return ComplexityLevel.MODERATE

    def _identify_required_capabilities(self, message: str) -> Set[RequiredCapability]:
        """Identify what capabilities are required for the request."""
        message_lower = message.lower()
        required = set()

        # Math keywords
        math_keywords = [
            "calculate",
            "compute",
            "add",
            "subtract",
            "multiply",
            "divide",
            "sum",
            "average",
            "percentage",
            "convert",
            "formula",
        ]
        if any(keyword in message_lower for keyword in math_keywords):
            required.add(RequiredCapability.BASIC_MATH)

        # Text processing keywords
        text_keywords = [
            "parse",
            "extract",
            "format",
            "process",
            "transform",
            "replace",
            "find",
            "search",
            "analyze text",
            "word count",
        ]
        if any(keyword in message_lower for keyword in text_keywords):
            required.add(RequiredCapability.TEXT_PROCESSING)

        # Information retrieval keywords
        info_keywords = [
            "search",
            "find information",
            "look up",
            "retrieve",
            "database",
            "query",
        ]
        if any(keyword in message_lower for keyword in info_keywords):
            required.add(RequiredCapability.INFORMATION_RETRIEVAL)

        # Memory keywords
        memory_keywords = [
            "remember",
            "recall",
            "what did",
            "earlier",
            "before",
            "previous",
            "history",
        ]
        if any(keyword in message_lower for keyword in memory_keywords):
            required.add(RequiredCapability.CONVERSATION_MEMORY)

        # Web search keywords
        web_keywords = [
            "current",
            "latest",
            "news",
            "recent",
            "today",
            "now",
            "real-time",
        ]
        if any(keyword in message_lower for keyword in web_keywords):
            required.add(RequiredCapability.WEB_SEARCH)

        # Summarization keywords
        summary_keywords = [
            "summarize",
            "summary",
            "brief",
            "overview",
            "condense",
            "key points",
        ]
        if any(keyword in message_lower for keyword in summary_keywords):
            required.add(RequiredCapability.SUMMARIZATION)

        # If no specific capabilities identified, assume general knowledge
        if not required:
            required.add(RequiredCapability.GENERAL_KNOWLEDGE)

        return required

    def _extract_computational_requirements(self, message: str) -> List[str]:
        """Extract specific computational requirements."""
        requirements = []
        message_lower = message.lower()

        # Algorithm requirements
        if any(
            word in message_lower
            for word in ["algorithm", "method", "procedure", "process"]
        ):
            requirements.append("custom_algorithm")

        # Data structure requirements
        if any(
            word in message_lower
            for word in ["array", "list", "matrix", "table", "database"]
        ):
            requirements.append("data_structures")

        # Mathematical functions
        if any(
            word in message_lower
            for word in ["function", "equation", "formula", "calculation"]
        ):
            requirements.append("mathematical_functions")

        # Iterative processing
        if any(
            word in message_lower
            for word in ["loop", "iterate", "repeat", "each", "every"]
        ):
            requirements.append("iterative_processing")

        # State management
        if any(
            word in message_lower
            for word in ["track", "monitor", "maintain", "remember", "state"]
        ):
            requirements.append("state_management")

        return requirements

    def _assess_domain_specificity(self, message: str) -> float:
        """Assess how domain-specific the request is (0-1)."""
        message_lower = message.lower()

        # Domain-specific keywords and their weights
        domain_indicators = {
            "financial": 0.8,
            "medical": 0.9,
            "legal": 0.9,
            "scientific": 0.7,
            "engineering": 0.7,
            "statistical": 0.6,
            "mathematical": 0.5,
            "programming": 0.6,
            "technical": 0.4,
            "specialized": 0.7,
            "proprietary": 0.8,
            "custom": 0.4,
            "specific": 0.3,
        }

        max_specificity = 0.0
        for keyword, weight in domain_indicators.items():
            if keyword in message_lower:
                max_specificity = max(max_specificity, weight)

        # Adjust based on jargon density
        words = message_lower.split()
        jargon_count = sum(
            1
            for word in words
            if len(word) > 8 or word.endswith(("tion", "ing", "ity", "ism"))
        )
        jargon_ratio = jargon_count / max(len(words), 1)

        return min(1.0, max_specificity + (jargon_ratio * 0.3))

    def _assess_reusability(self, message: str) -> float:
        """Assess how reusable a potential tool would be (0-1)."""
        message_lower = message.lower()

        # High reusability indicators
        reusable_patterns = [
            r"\bconvert\b.*\bto\b",  # Conversion functions
            r"\bcalculate\b.*\b(percentage|interest|tax|discount)",  # Common calculations
            r"\bformat\b.*\b(text|data|number)",  # Formatting functions
            r"\bparse\b.*\b(json|xml|csv)",  # Parsing functions
            r"\bvalidate\b.*\b(email|phone|url)",  # Validation functions
        ]

        reusability_score = 0.5  # Base score

        # Increase for generic operation words
        generic_words = [
            "calculate",
            "convert",
            "format",
            "parse",
            "validate",
            "process",
            "transform",
        ]
        for word in generic_words:
            if word in message_lower:
                reusability_score += 0.1

        # Decrease for very specific details
        specific_indicators = ["this", "my", "today", "now", "here", "me", "I"]
        for indicator in specific_indicators:
            if indicator.lower() in message_lower:
                reusability_score -= 0.1

        # Check for reusable patterns
        for pattern in reusable_patterns:
            if re.search(pattern, message_lower):
                reusability_score += 0.2

        return max(0.0, min(1.0, reusability_score))

    def _extract_primary_intent(self, message: str) -> str:
        """Extract the primary intent from the message."""
        message_lower = message.lower().strip()

        # Intent patterns
        intent_patterns = {
            "calculation": r"\b(calculate|compute|add|subtract|multiply|divide|sum)\b",
            "conversion": r"\b(convert|transform|change)\b.*\bto\b",
            "analysis": r"\b(analyze|examine|study|investigate)\b",
            "processing": r"\b(process|handle|manage|organize)\b",
            "information": r"\b(tell|explain|describe|what|how|why)\b",
            "search": r"\b(find|search|look|locate)\b",
            "creation": r"\b(create|make|build|generate|produce)\b",
            "formatting": r"\b(format|style|arrange|organize)\b",
            "validation": r"\b(check|verify|validate|confirm)\b",
        }

        for intent, pattern in intent_patterns.items():
            if re.search(pattern, message_lower):
                return intent

        return "general_query"

    def _calculate_confidence(
        self,
        message: str,
        complexity: ComplexityLevel,
        capabilities: Set[RequiredCapability],
        requirements: List[str],
    ) -> float:
        """Calculate confidence in the analysis."""
        base_confidence = 0.7

        # Adjust based on message length and clarity
        words = message.split()
        if len(words) < 3:
            base_confidence -= 0.2  # Very short messages are ambiguous
        elif len(words) > 50:
            base_confidence -= 0.1  # Very long messages may be unclear

        # Adjust based on complexity clarity
        if complexity in [ComplexityLevel.TRIVIAL, ComplexityLevel.SPECIALIZED]:
            base_confidence += 0.1  # Clear extremes

        # Adjust based on requirement specificity
        if len(requirements) > 0:
            base_confidence += 0.1

        # Adjust based on capability identification
        if len(capabilities) == 1:
            base_confidence += 0.1  # Clear single capability
        elif len(capabilities) > 4:
            base_confidence -= 0.1  # Too many capabilities may indicate confusion

        return max(0.1, min(1.0, base_confidence))

    def should_generate_tool(self, analysis: IntentAnalysis) -> Tuple[bool, str]:
        """Determine if a dynamic tool should be generated based on analysis."""

        # Strong "NO" conditions (return False immediately)
        if analysis.complexity_level == ComplexityLevel.TRIVIAL:
            return False, "Request is too simple for a dynamic tool"

        if analysis.confidence < 0.4:
            return False, "Analysis confidence too low to proceed"

        if (
            len(analysis.required_capabilities) == 1
            and RequiredCapability.GENERAL_KNOWLEDGE in analysis.required_capabilities
        ):
            return False, "Request can be handled with general knowledge"

        # Check if existing capabilities can handle the request
        if self._can_handle_with_existing_capabilities(analysis):
            return False, "Existing system capabilities are sufficient"

        # Strong "YES" conditions
        if analysis.complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.SPECIALIZED,
        ]:
            if analysis.reusability_potential > 0.6:
                return (
                    True,
                    f"Complex request with high reusability: {analysis.primary_intent}",
                )

        if analysis.domain_specificity > 0.7:
            return (
                True,
                f"Domain-specific request requiring specialized logic: {analysis.primary_intent}",
            )

        if len(analysis.computational_requirements) >= 2:
            return (
                True,
                f"Multiple computational requirements: {', '.join(analysis.computational_requirements)}",
            )

        # Moderate conditions (require higher thresholds)
        if analysis.complexity_level == ComplexityLevel.MODERATE:
            if analysis.reusability_potential > 0.7 and analysis.confidence > 0.6:
                return (
                    True,
                    f"Moderately complex with high reusability: {analysis.primary_intent}",
                )

        # Default to no tool needed
        return False, "Request can likely be handled without a dynamic tool"

    def _can_handle_with_existing_capabilities(self, analysis: IntentAnalysis) -> bool:
        """Check if existing system capabilities can handle the request."""

        # For each required capability, check if system limitations are hit
        for capability in analysis.required_capabilities:
            if capability not in self.system_capabilities:
                continue  # Unknown capability, assume tool needed

            capability_info = self.system_capabilities[capability]

            # Check if computational requirements exceed capability limitations
            for requirement in analysis.computational_requirements:
                if requirement in capability_info.get("limitations", []):
                    return False  # System limitation hit, tool needed

        # If complexity is moderate or higher and involves multiple capabilities
        if (
            analysis.complexity_level
            in [
                ComplexityLevel.MODERATE,
                ComplexityLevel.COMPLEX,
                ComplexityLevel.SPECIALIZED,
            ]
            and len(analysis.required_capabilities) > 2
        ):
            return False  # Multiple complex capabilities likely need integration

        # If domain specificity is high, likely need specialized tool
        if analysis.domain_specificity > 0.6:
            return False

        return True  # Existing capabilities should suffice


# Global analyzer instance
smart_analyzer = SmartIntentAnalyzer()
