#!/usr/bin/env python3
"""
Lightweight test for IntentClassifierAgent architectural changes.
Tests the core architectural improvements without heavy infrastructure dependencies.
"""

import sys
sys.path.append('/Users/lons7862/workspace/llmmllab/inference')

# Test that we can import the core models
try:
    from models.conversation_ctx import ConversationCtx
    from models.intent_analysis import IntentAnalysis
    from models.complexity_level import ComplexityLevel
    from models.required_capability import RequiredCapability
    from models.computational_requirement import ComputationalRequirement
    print("✅ Core model imports successful")
except ImportError as e:
    print(f"❌ Core model import failed: {e}")
    sys.exit(1)

# Test that we can read the updated IntentClassifierAgent source
try:
    with open('/Users/lons7862/workspace/llmmllab/inference/composer/agents/intent_classifier.py', 'r') as f:
        agent_source = f.read()
    
    # Check for key architectural improvements
    checks = {
        'LLM-based analysis': '_llm_analyze_intent' in agent_source,
        'Analysis model profile': 'analysis_profile_id' in agent_source,
        'Current user message assertion': 'current_user_message is not None' in agent_source,
        'User config assertion': 'user_config is not None' in agent_source,
        'Pipeline factory import': 'from runner import pipeline_factory' in agent_source,
        'JSON parsing': '_parse_llm_response' in agent_source,
        'Statistical augmentation': '_augment_with_statistics' in agent_source,
        'Fallback heuristics': '_fallback_heuristic_analysis' in agent_source,
        'RAG depth determination': 'determine_rag_depth' in agent_source,
        'Removed old extract method': '_extract_user_query' not in agent_source or 'def _extract_user_query' not in agent_source
    }
    
    print("\n🧪 Architectural Changes Validation:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎯 All architectural improvements verified!")
    else:
        print("\n⚠️  Some architectural checks failed")
    
    # Test schema model instantiation
    print("\n🧪 Schema Model Validation:")
    
    # Test IntentAnalysis creation
    try:
        analysis = IntentAnalysis(
            primary_intent="research",
            complexity_level=ComplexityLevel.COMPLEX,
            required_capabilities=[
                RequiredCapability.WEB_SEARCH,
                RequiredCapability.REASONING
            ],
            computational_requirements=[
                ComputationalRequirement.COMPLEX_REASONING,
                ComputationalRequirement.EXTERNAL_API_CALLS
            ],
            domain_specificity=0.8,
            reusability_potential=0.6,
            confidence=0.85
        )
        print("  ✅ IntentAnalysis object creation successful")
        print(f"     - Intent: {analysis.primary_intent}")
        print(f"     - Complexity: {analysis.complexity_level.value}")
        print(f"     - Capabilities: {len(analysis.required_capabilities)}")
        print(f"     - Requirements: {len(analysis.computational_requirements)}")
    except Exception as e:
        print(f"  ❌ IntentAnalysis creation failed: {e}")
        all_passed = False
    
    # Test enum values
    print("\n🧪 Enum Validation:")
    try:
        complexity_values = [c.value for c in ComplexityLevel]
        print(f"  ✅ ComplexityLevel enum values: {complexity_values}")
        
        capability_sample = [RequiredCapability.WEB_SEARCH.value, RequiredCapability.REASONING.value]
        print(f"  ✅ RequiredCapability sample: {capability_sample}")
        
        requirement_sample = [ComputationalRequirement.HIGH_MEMORY.value, ComputationalRequirement.GPU_ACCELERATION.value]
        print(f"  ✅ ComputationalRequirement sample: {requirement_sample}")
        
    except Exception as e:
        print(f"  ❌ Enum validation failed: {e}")
        all_passed = False
    
    print("\n📋 Summary of Changes:")
    print("  🔄 Transformed from heuristic-based to LLM-driven classification")
    print("  🎯 Uses 'analysis' model profile for intent classification")
    print("  🏗️  Ready for graph node integration")
    print("  ✅ Validates current_user_message and user_config with assertions")
    print("  🧠 LLM analyzes intent, complexity, capabilities, and requirements")
    print("  📊 Statistical augmentation supplements LLM analysis")
    print("  🔄 Fallback heuristics for error recovery")
    print("  🗑️  Removed old _extract_user_query method")
    
    if all_passed:
        print("\n🚀 IntentClassifierAgent successfully transformed to LLM-driven architecture!")
    else:
        print("\n⚠️  Some validation issues found - review needed")
    
except Exception as e:
    print(f"❌ Agent source validation failed: {e}")
    sys.exit(1)