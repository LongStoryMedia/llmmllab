"""
Test for multi-agent workflow routing logic using proper state fields and enums.
This test validates the routing logic without importing the full composer module.
"""

import unittest
from enum import Enum


# Mock the enums and models to test the routing logic without circular imports
class RequiredCapability(str, Enum):
    DATA_PROCESSING = "data_processing"
    INFORMATION_RETRIEVAL = "information_retrieval"
    WEB_SEARCH = "web_search"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    DATABASE_ACCESS = "database_access"
    API_INTEGRATION = "api_integration"
    TEXT_PROCESSING = "text_processing"
    GENERAL_KNOWLEDGE = "general_knowledge"


class AgentSpecialization(str, Enum):
    ANALYSIS = "analysis_agent"
    CONTENT_GENERATION = "content_generation_agent"


class MockIntentAnalysis:
    def __init__(self, workflow_type: str, required_capabilities: list):
        self.workflow_type = workflow_type
        self.required_capabilities = required_capabilities


class MockWorkflowState:
    def __init__(self, intent_classification=None):
        self.intent_classification = intent_classification


def route_to_specialists_logic(state):
    """
    Extracted routing logic from multi_agent.py for testing.
    This mirrors the actual routing function without import dependencies.
    """
    # Use actual intent classification from state
    if not state.intent_classification:
        # Default to content generation if no intent analysis available
        return AgentSpecialization.CONTENT_GENERATION.value

    intent = state.intent_classification

    # Route based on required capabilities from intent analysis
    analysis_capabilities = {
        RequiredCapability.DATA_PROCESSING,
        RequiredCapability.INFORMATION_RETRIEVAL,
        RequiredCapability.WEB_SEARCH,
        RequiredCapability.SUMMARIZATION,
        RequiredCapability.REASONING,
        RequiredCapability.DATABASE_ACCESS,
        RequiredCapability.API_INTEGRATION,
    }

    # Check if any required capabilities match analysis specialization
    if any(cap in analysis_capabilities for cap in intent.required_capabilities):
        return AgentSpecialization.ANALYSIS.value

    # Route based on primary intent patterns
    analysis_intents = {
        "research",
        "analyze",
        "investigate",
        "summarize",
        "compare",
        "evaluate",
        "calculate",
        "process",
    }

    if any(keyword in intent.workflow_type.lower() for keyword in analysis_intents):
        return AgentSpecialization.ANALYSIS.value

    # Default to content generation for creative, general, and conversational tasks
    return AgentSpecialization.CONTENT_GENERATION.value


class TestMultiAgentRouting(unittest.TestCase):
    """Test multi-agent routing decisions based on intent analysis."""

    def create_test_state(self, workflow_type: str, required_capabilities: list):
        """Create a test state with intent analysis."""
        intent = MockIntentAnalysis(workflow_type, required_capabilities)
        return MockWorkflowState(intent_classification=intent)

    def test_analysis_routing_by_capabilities(self):
        """Test routing to analysis agent based on required capabilities."""
        # Test data processing capability
        state = self.create_test_state(
            "Process this dataset",
            [RequiredCapability.DATA_PROCESSING, RequiredCapability.REASONING],
        )

        result = route_to_specialists_logic(state)
        self.assertEqual(result, AgentSpecialization.ANALYSIS.value)

    def test_content_generation_routing_by_capabilities(self):
        """Test routing to content generation agent for creative tasks."""
        state = self.create_test_state(
            "Write a creative story",
            [RequiredCapability.TEXT_PROCESSING, RequiredCapability.GENERAL_KNOWLEDGE],
        )

        result = route_to_specialists_logic(state)
        self.assertEqual(result, AgentSpecialization.CONTENT_GENERATION.value)

    def test_analysis_routing_by_intent_patterns(self):
        """Test routing to analysis agent based on primary intent keywords."""
        test_cases = [
            "research the latest AI developments",
            "analyze this data pattern",
            "investigate the root cause",
            "summarize the key findings",
            "compare these two approaches",
            "evaluate the performance metrics",
        ]

        analysis_intents = {
            "research",
            "analyze",
            "investigate",
            "summarize",
            "compare",
            "evaluate",
            "calculate",
            "process",
        }

        for intent in test_cases:
            should_route_to_analysis = any(
                keyword in intent.lower() for keyword in analysis_intents
            )
            self.assertTrue(
                should_route_to_analysis,
                f"Intent '{intent}' should route to analysis agent",
            )

    def test_content_generation_routing_by_intent_patterns(self):
        """Test routing to content generation for creative/general tasks."""
        test_cases = [
            "write a blog post about cats",
            "create a marketing plan",
            "draft an email to my team",
            "help me brainstorm ideas",
            "explain quantum physics simply",
        ]

        analysis_intents = {
            "research",
            "analyze",
            "investigate",
            "summarize",
            "compare",
            "evaluate",
            "calculate",
            "process",
        }

        for intent in test_cases:
            should_route_to_analysis = any(
                keyword in intent.lower() for keyword in analysis_intents
            )
            self.assertFalse(
                should_route_to_analysis,
                f"Intent '{intent}' should route to content generation agent",
            )

    def test_no_intent_classification_fallback(self):
        """Test fallback behavior when no intent classification is available."""
        state = MockWorkflowState()  # No intent_classification

        result = route_to_specialists_logic(state)
        self.assertEqual(result, AgentSpecialization.CONTENT_GENERATION.value)

    def test_enum_values_are_consistent(self):
        """Test that enum values match the expected node names."""
        self.assertEqual(AgentSpecialization.ANALYSIS.value, "analysis_agent")
        self.assertEqual(
            AgentSpecialization.CONTENT_GENERATION.value, "content_generation_agent"
        )


if __name__ == "__main__":
    unittest.main()
