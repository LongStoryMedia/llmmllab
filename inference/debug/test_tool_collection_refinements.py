"""
Test script for refined tool collection node with enhanced intent-based filtering.
Tests the new IntentAnalysis properties and improved decision logic.
"""

from models import (
    IntentAnalysis, 
    ComputationalRequirement, 
    ComplexityLevel, 
    WorkflowType, 
    RequiredCapability,
    Tool
)


def test_intent_analysis_properties():
    """Test that new IntentAnalysis properties work correctly."""
    print("🧪 Testing IntentAnalysis with new properties...")
    
    # Test dynamic tool requirement case
    intent = IntentAnalysis(
        workflow_type=WorkflowType.ENGINEERING,
        complexity_level=ComplexityLevel.COMPLEX,
        required_capabilities=[RequiredCapability.API_INTEGRATION, RequiredCapability.DATA_PROCESSING],
        computational_requirements=ComputationalRequirement.HIGH,
        domain_specificity=0.9,
        reusability_potential=0.3,
        confidence=0.85,
        requires_tools=True,
        requires_custom_tools=True,
        tool_complexity_score=0.8
    )
    
    print(f"✅ Dynamic tool case:")
    print(f"   requires_tools: {intent.requires_tools}")
    print(f"   requires_custom_tools: {intent.requires_custom_tools}")
    print(f"   tool_complexity_score: {intent.tool_complexity_score}")
    print(f"   computational_requirements: {intent.computational_requirements.value}")
    
    # Test simple case
    simple_intent = IntentAnalysis(
        workflow_type=WorkflowType.GENERAL,
        complexity_level=ComplexityLevel.SIMPLE, 
        required_capabilities=[RequiredCapability.INFORMATION_RETRIEVAL],
        computational_requirements=ComputationalRequirement.LOW,
        domain_specificity=0.2,
        reusability_potential=0.8,
        confidence=0.95,
        requires_tools=True,
        requires_custom_tools=False,
        tool_complexity_score=0.3
    )
    
    print(f"✅ Simple tool case:")
    print(f"   requires_tools: {simple_intent.requires_tools}")
    print(f"   requires_custom_tools: {simple_intent.requires_custom_tools}")
    print(f"   tool_complexity_score: {simple_intent.tool_complexity_score}")
    
    return intent, simple_intent


def test_dynamic_tool_decision_logic():
    """Test the enhanced dynamic tool decision logic."""
    from composer.nodes.tools.tool_collection import ToolCollectionNode
    from composer.tools.registry import ToolRegistry
    from composer.agents.engineering_agent import EngineeringAgent
    
    print("\n🧪 Testing dynamic tool decision logic...")
    
    # Mock dependencies - in real usage these would be injected
    mock_registry = None  # We'll test the logic without full setup
    mock_agent = None
    
    # Create node instance
    node = ToolCollectionNode(mock_registry, mock_agent)
    
    # Test case 1: Should generate dynamic tools (custom tools required)
    intent1 = IntentAnalysis(
        workflow_type=WorkflowType.ENGINEERING,
        complexity_level=ComplexityLevel.COMPLEX,
        required_capabilities=[RequiredCapability.API_INTEGRATION],
        computational_requirements=ComputationalRequirement.HIGH,
        domain_specificity=0.9,
        reusability_potential=0.3,
        confidence=0.85,
        requires_tools=True,
        requires_custom_tools=True,
        tool_complexity_score=0.8
    )
    
    user_config = type('MockConfig', (), {
        'tool': type('MockToolConfig', (), {'enable_tool_generation': True})()
    })()
    
    should_generate = node._should_generate_dynamic_tools([intent1], user_config)
    print(f"✅ Custom tools required case: {should_generate} (expected: True)")
    
    # Test case 2: Should not generate dynamic tools (simple case)
    intent2 = IntentAnalysis(
        workflow_type=WorkflowType.GENERAL,
        complexity_level=ComplexityLevel.SIMPLE,
        required_capabilities=[RequiredCapability.INFORMATION_RETRIEVAL],
        computational_requirements=ComputationalRequirement.LOW,
        domain_specificity=0.2,
        reusability_potential=0.8,
        confidence=0.95,
        requires_tools=True,
        requires_custom_tools=False,
        tool_complexity_score=0.3
    )
    
    should_not_generate = node._should_generate_dynamic_tools([intent2], user_config)
    print(f"✅ Simple tools case: {should_not_generate} (expected: False)")
    
    # Test case 3: High complexity + high tool complexity should generate
    intent3 = IntentAnalysis(
        workflow_type=WorkflowType.RESEARCH,
        complexity_level=ComplexityLevel.SPECIALIZED,
        required_capabilities=[RequiredCapability.DATA_PROCESSING, RequiredCapability.WEB_SEARCH],
        computational_requirements=ComputationalRequirement.INTENSIVE,
        domain_specificity=0.85,
        reusability_potential=0.4,
        confidence=0.9,
        requires_tools=True,
        requires_custom_tools=False,
        tool_complexity_score=0.9
    )
    
    should_generate_complex = node._should_generate_dynamic_tools([intent3], user_config)
    print(f"✅ High complexity case: {should_generate_complex} (expected: True)")


def test_static_tool_filtering():
    """Test the intent-based static tool filtering."""
    from composer.nodes.tools.tool_collection import ToolCollectionNode
    
    print("\n🧪 Testing static tool filtering logic...")
    
    node = ToolCollectionNode(None, None)
    
    # Mock tools
    search_tool = type('MockTool', (), {'name': 'web_search'})()
    memory_tool = type('MockTool', (), {'name': 'memory_search'})()
    api_tool = type('MockTool', (), {'name': 'api_integration'})()
    math_tool = type('MockTool', (), {'name': 'basic_calculator'})()
    
    # Test intent requiring web search
    search_intent = IntentAnalysis(
        workflow_type=WorkflowType.RESEARCH,
        complexity_level=ComplexityLevel.MODERATE,
        required_capabilities=[RequiredCapability.WEB_SEARCH, RequiredCapability.INFORMATION_RETRIEVAL],
        computational_requirements=ComputationalRequirement.MODERATE,
        domain_specificity=0.5,
        reusability_potential=0.7,
        confidence=0.9,
        requires_tools=True,
        requires_custom_tools=False,
        tool_complexity_score=0.4
    )
    
    # Test tool matching
    should_include_search = node._should_include_static_tool(search_tool, [search_intent], None)
    should_include_memory = node._should_include_static_tool(memory_tool, [search_intent], None)
    should_include_api = node._should_include_static_tool(api_tool, [search_intent], None)
    
    print(f"✅ Web search tool for search intent: {should_include_search} (expected: True)")
    print(f"✅ Memory tool for search intent: {should_include_memory} (expected: False)")
    print(f"✅ API tool for search intent: {should_include_api} (expected: False)")
    
    # Test math intent
    math_intent = IntentAnalysis(
        workflow_type=WorkflowType.GENERAL,
        complexity_level=ComplexityLevel.SIMPLE,
        required_capabilities=[RequiredCapability.BASIC_MATH],
        computational_requirements=ComputationalRequirement.LOW,
        domain_specificity=0.1,
        reusability_potential=0.9,
        confidence=0.95,
        requires_tools=True,
        requires_custom_tools=False,
        tool_complexity_score=0.2
    )
    
    should_include_math = node._should_include_static_tool(math_tool, [math_intent], None)
    print(f"✅ Math tool for math intent: {should_include_math} (expected: True)")


if __name__ == "__main__":
    print("🚀 Testing refined tool collection node...")
    
    test_intent_analysis_properties()
    test_dynamic_tool_decision_logic()
    test_static_tool_filtering()
    
    print("\n✅ All tool collection refinement tests completed!")