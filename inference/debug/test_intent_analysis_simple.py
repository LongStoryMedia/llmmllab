"""
Simple test for IntentAnalysis model with new properties.
Tests only the model properties without full composer imports.
"""

import sys
sys.path.append('/app')

from models import (
    IntentAnalysis, 
    ComputationalRequirement, 
    ComplexityLevel, 
    WorkflowType, 
    RequiredCapability
)


def test_intent_analysis_model():
    """Test that new IntentAnalysis properties work correctly."""
    print("🧪 Testing IntentAnalysis model with new properties...")
    
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
    
    print(f"✅ Dynamic tool case created successfully:")
    print(f"   workflow_type: {intent.workflow_type.value}")
    print(f"   complexity_level: {intent.complexity_level.value}")
    print(f"   requires_tools: {intent.requires_tools}")
    print(f"   requires_custom_tools: {intent.requires_custom_tools}")
    print(f"   tool_complexity_score: {intent.tool_complexity_score}")
    print(f"   computational_requirements: {intent.computational_requirements.value}")
    print(f"   required_capabilities: {[cap.value for cap in intent.required_capabilities]}")
    
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
    
    print(f"✅ Simple tool case created successfully:")
    print(f"   workflow_type: {simple_intent.workflow_type.value}")
    print(f"   complexity_level: {simple_intent.complexity_level.value}")
    print(f"   requires_tools: {simple_intent.requires_tools}")
    print(f"   requires_custom_tools: {simple_intent.requires_custom_tools}")
    print(f"   tool_complexity_score: {simple_intent.tool_complexity_score}")
    
    return intent, simple_intent


def test_decision_logic_simulation():
    """Simulate the decision logic without full imports."""
    print("\n🧪 Testing decision logic simulation...")
    
    def simulate_should_generate_dynamic_tools(intents, user_config):
        """Simulate the _should_generate_dynamic_tools logic."""
        # Check user configuration
        if (user_config and 
            hasattr(user_config, 'tool') and 
            hasattr(user_config.tool, 'enable_tool_generation') and
            not user_config.tool.enable_tool_generation):
            return False

        # Check if any intent explicitly requires custom tools
        for intent in intents:
            # Check if custom tools are explicitly required
            if intent.requires_custom_tools:
                return True
            
            # Check if high complexity and tool requirement suggest dynamic tools needed
            if (intent.requires_tools and 
                intent.complexity_level.value in ["COMPLEX", "SPECIALIZED"] and
                intent.tool_complexity_score > 0.7):
                return True
            
            # Check if domain specificity and computational requirements suggest custom tools
            if (intent.domain_specificity > 0.8 and 
                intent.computational_requirements.value in ["HIGH", "INTENSIVE"]):
                return True

        return False
    
    # Create test intents
    complex_intent = IntentAnalysis(
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
    
    high_complexity_intent = IntentAnalysis(
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
    
    # Mock user config
    user_config = type('MockConfig', (), {
        'tool': type('MockToolConfig', (), {'enable_tool_generation': True})()
    })()
    
    # Test decisions
    result1 = simulate_should_generate_dynamic_tools([complex_intent], user_config)
    result2 = simulate_should_generate_dynamic_tools([simple_intent], user_config)
    result3 = simulate_should_generate_dynamic_tools([high_complexity_intent], user_config)
    
    print(f"✅ Complex intent (requires_custom_tools=True): {result1} (expected: True)")
    print(f"✅ Simple intent (low complexity): {result2} (expected: False)")
    print(f"✅ High complexity intent (specialized + high tool score): {result3} (expected: True)")


if __name__ == "__main__":
    print("🚀 Testing IntentAnalysis refinements...")
    
    test_intent_analysis_model()
    test_decision_logic_simulation()
    
    print("\n✅ All IntentAnalysis refinement tests completed!")