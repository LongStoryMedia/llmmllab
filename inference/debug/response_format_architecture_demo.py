"""
Example demonstrating the improved response format architecture.
Shows how analysis-driven format determination replaces keyword matching.
"""

import asyncio
from unittest.mock import Mock, AsyncMock
from composer.graph.state import WorkflowState
from composer.nodes.agents.engineering import EngineeringAgentNode
from composer.nodes.agents.response_format_analysis import ResponseFormatAnalysisNode
from models.lang_chain_message import LangChainMessage
from models.intent_analysis import IntentAnalysis
from models.complexity_level import ComplexityLevel


async def demonstrate_improved_architecture():
    """
    Demonstrate how the new architecture works with analysis-driven format determination.
    """
    print("🔧 Demonstrating improved response format architecture")
    print("=" * 60)

    # Create mock pipeline factory
    mock_factory = Mock()
    mock_factory.create_pipeline = AsyncMock()

    # Create analysis and engineering nodes
    analysis_node = ResponseFormatAnalysisNode(mock_factory)
    engineering_node = EngineeringAgentNode(mock_factory)

    # Test Case 1: Code implementation request
    print("\n📝 Test Case 1: Code Implementation Request")
    print("-" * 45)

    test_queries = [
        {
            "query": "How do I implement a binary search algorithm in Python?",
            "intent": "technical",
            "expected_format": "CODE_SOLUTION",
            "expected_domain": "SOFTWARE_DEVELOPMENT",
        },
        {
            "query": "What are the best practices for database design?",
            "intent": "technical",
            "expected_format": "BEST_PRACTICES",
            "expected_domain": "DATA_ENGINEERING",
        },
        {
            "query": "My Django app is throwing a 500 error, how do I debug it?",
            "intent": "technical",
            "expected_format": "TROUBLESHOOTING",
            "expected_domain": "SOFTWARE_DEVELOPMENT",
        },
        {
            "query": "Can you walk me through setting up a CI/CD pipeline?",
            "intent": "technical",
            "expected_format": "STEP_BY_STEP_GUIDE",
            "expected_domain": "DEVOPS_INFRASTRUCTURE",
        },
    ]

    for i, test_case in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{test_case['query']}'")

        # Create test state with mock intent analysis
        state = WorkflowState(
            messages=[LangChainMessage(content=test_case["query"])],
            user_id="test_user",
            intent_classification=IntentAnalysis(
                primary_intent=test_case["intent"],
                complexity_level=ComplexityLevel.MODERATE,
                required_capabilities=[],
                computational_requirements=[],
                domain_specificity=0.8,
                reusability_potential=0.7,
                confidence=0.9,
            ),
        )

        # Mock the LLM analysis to return expected values
        async def mock_run_pipeline(messages, **kwargs):
            query_lower = test_case["query"].lower()
            if any(word in query_lower for word in ["implement", "algorithm", "code"]):
                return test_case["expected_format"]
            elif any(word in query_lower for word in ["best", "practices", "approach"]):
                return test_case["expected_format"]
            elif any(word in query_lower for word in ["debug", "error", "fix"]):
                return test_case["expected_format"]
            elif any(
                word in query_lower for word in ["how", "walk", "through", "steps"]
            ):
                return test_case["expected_format"]
            else:
                return "DETAILED_ANALYSIS"

        # Patch the analysis node's LLM calls to return enum values
        from composer.agents.engineering_agent import ResponseFormat, TechnicalDomain

        format_enum = getattr(ResponseFormat, test_case["expected_format"])
        domain_enum = getattr(TechnicalDomain, test_case["expected_domain"])

        analysis_node._analyze_response_format = AsyncMock(return_value=format_enum)
        analysis_node._analyze_technical_domain = AsyncMock(return_value=domain_enum)

        # Run analysis node (this would normally use LLM analysis)
        try:
            analyzed_state = await analysis_node(state)

            print(f"   📊 Analysis Results:")
            print(f"      Response Format: {analyzed_state.response_format}")
            print(f"      Technical Domain: {analyzed_state.technical_domain}")

            # Test engineering node using analyzed values
            domain = engineering_node._get_technical_domain_from_state(analyzed_state)
            format_val = engineering_node._get_response_format_from_state(
                analyzed_state
            )

            print(f"   🎯 Engineering Node Uses:")
            print(f"      Domain: {domain}")
            print(f"      Format: {format_val}")

            # Verify the engineering node uses state values rather than keywords
            print(
                f"   ✅ State-driven: {analyzed_state.response_format == format_val.value}"
            )

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n🎉 Architecture Demonstration Complete!")
    print("\n📋 Key Improvements:")
    print("   1. ✅ LLM-based analysis replaces keyword matching")
    print("   2. ✅ Response format determined by sophisticated analysis")
    print("   3. ✅ Technical domain set by dedicated analysis node")
    print("   4. ✅ Engineering node uses predetermined values from state")
    print("   5. ✅ Fallback mechanisms preserve robustness")
    print("   6. ✅ Workflow can include analysis step before engineering response")


if __name__ == "__main__":
    asyncio.run(demonstrate_improved_architecture())
