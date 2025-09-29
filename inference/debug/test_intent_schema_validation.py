#!/usr/bin/env python3
"""
Simple test script to validate IntentAnalysis schema without heavy dependencies.
Tests that the schema models can be instantiated and used correctly.
"""

import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

from models.intent_analysis import IntentAnalysis
from models.complexity_level import ComplexityLevel
from models.required_capability import RequiredCapability
from models.computational_requirement import ComputationalRequirement


def test_intent_analysis_schema():
    """Test that IntentAnalysis schema works correctly with all enum types."""
    
    print("🧪 Testing IntentAnalysis Schema Validation\n")
    
    # Test 1: Create a basic IntentAnalysis object
    print("Test 1: Creating basic IntentAnalysis object")
    try:
        analysis = IntentAnalysis(
            primary_intent="chat",
            complexity_level=ComplexityLevel.TRIVIAL,
            required_capabilities=[RequiredCapability.TEXT_PROCESSING],
            computational_requirements=[],
            domain_specificity=0.1,
            reusability_potential=0.9,
            confidence=0.95
        )
        print(f"  ✅ Successfully created: {analysis.primary_intent} with complexity {analysis.complexity_level.value}")
    except Exception as e:
        print(f"  ❌ Failed to create basic analysis: {e}")
    
    # Test 2: Create complex research analysis
    print("\nTest 2: Creating complex research analysis")
    try:
        analysis = IntentAnalysis(
            primary_intent="research",
            complexity_level=ComplexityLevel.COMPLEX,
            required_capabilities=[
                RequiredCapability.WEB_SEARCH,
                RequiredCapability.REASONING,
                RequiredCapability.INFORMATION_RETRIEVAL
            ],
            computational_requirements=[
                ComputationalRequirement.HIGH_MEMORY,
                ComputationalRequirement.COMPLEX_REASONING
            ],
            domain_specificity=0.8,
            reusability_potential=0.3,
            confidence=0.87
        )
        print(f"  ✅ Successfully created research analysis:")
        print(f"     - Intent: {analysis.primary_intent}")
        print(f"     - Complexity: {analysis.complexity_level.value}")
        print(f"     - Capabilities: {[cap.value for cap in analysis.required_capabilities]}")
        print(f"     - Requirements: {[req.value for req in analysis.computational_requirements]}")
    except Exception as e:
        print(f"  ❌ Failed to create research analysis: {e}")
    
    # Test 3: Test all primary intent values
    print("\nTest 3: Validating all primary intent enum values")
    intent_values = ["chat", "research", "creative", "technical", "summarization", "analysis", "tool_use"]
    
    for intent in intent_values:
        try:
            analysis = IntentAnalysis(
                primary_intent=intent,
                complexity_level=ComplexityLevel.SIMPLE,
                required_capabilities=[RequiredCapability.TEXT_PROCESSING],
                computational_requirements=[],
                domain_specificity=0.5,
                reusability_potential=0.5,
                confidence=0.8
            )
            print(f"  ✅ Intent '{intent}' validated successfully")
        except Exception as e:
            print(f"  ❌ Intent '{intent}' validation failed: {e}")
    
    # Test 4: Test all complexity levels
    print("\nTest 4: Validating all complexity levels")
    complexity_levels = [
        ComplexityLevel.TRIVIAL,
        ComplexityLevel.SIMPLE,
        ComplexityLevel.MODERATE,
        ComplexityLevel.COMPLEX,
        ComplexityLevel.SPECIALIZED
    ]
    
    for complexity in complexity_levels:
        try:
            analysis = IntentAnalysis(
                primary_intent="technical",
                complexity_level=complexity,
                required_capabilities=[RequiredCapability.REASONING],
                computational_requirements=[],
                domain_specificity=0.6,
                reusability_potential=0.4,
                confidence=0.85
            )
            print(f"  ✅ Complexity '{complexity.value}' validated successfully")
        except Exception as e:
            print(f"  ❌ Complexity '{complexity.value}' validation failed: {e}")
    
    # Test 5: Test computational requirements
    print("\nTest 5: Validating computational requirements")
    comp_requirements = [
        ComputationalRequirement.HIGH_MEMORY,
        ComputationalRequirement.GPU_ACCELERATION,
        ComputationalRequirement.PARALLEL_PROCESSING,
        ComputationalRequirement.REAL_TIME_PROCESSING,
        ComputationalRequirement.LARGE_DATA_HANDLING,
        ComputationalRequirement.COMPLEX_REASONING
    ]
    
    try:
        analysis = IntentAnalysis(
            primary_intent="technical",
            complexity_level=ComplexityLevel.SPECIALIZED,
            required_capabilities=[RequiredCapability.REASONING, RequiredCapability.DATA_PROCESSING],
            computational_requirements=comp_requirements,
            domain_specificity=0.9,
            reusability_potential=0.1,
            confidence=0.75
        )
        print(f"  ✅ All computational requirements validated:")
        for req in comp_requirements:
            print(f"     - {req.value}")
    except Exception as e:
        print(f"  ❌ Computational requirements validation failed: {e}")
    
    # Test 6: Test required capabilities
    print("\nTest 6: Validating required capabilities")
    capabilities = [
        RequiredCapability.TEXT_PROCESSING,
        RequiredCapability.WEB_SEARCH,
        RequiredCapability.REASONING,
        RequiredCapability.INFORMATION_RETRIEVAL,
        RequiredCapability.SUMMARIZATION,
        RequiredCapability.DATA_PROCESSING,
        RequiredCapability.API_INTEGRATION,
        RequiredCapability.FILE_MANIPULATION
    ]
    
    try:
        analysis = IntentAnalysis(
            primary_intent="creative",
            complexity_level=ComplexityLevel.MODERATE,
            required_capabilities=capabilities,
            computational_requirements=[ComputationalRequirement.HIGH_MEMORY],
            domain_specificity=0.7,
            reusability_potential=0.6,
            confidence=0.88
        )
        print(f"  ✅ All required capabilities validated:")
        for cap in capabilities:
            print(f"     - {cap.value}")
    except Exception as e:
        print(f"  ❌ Required capabilities validation failed: {e}")
    
    print("\n🎯 IntentAnalysis Schema Validation Complete")


if __name__ == "__main__":
    test_intent_analysis_schema()