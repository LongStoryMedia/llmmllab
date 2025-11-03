#!/usr/bin/env python3
"""
Debug script to trace intent analysis flow through the workflow.
Tests the exact path from planning_intent subgraph to engineering node.
"""

import asyncio
from models import UserConfig, LangChainMessage, IntentAnalysis, WorkflowType, TechnicalDomain, ResponseFormat

from utils.logging import llmmllogger

logger = llmmllogger.bind(component="WorkflowIntentFlowDebug")

async def debug_workflow_intent_flow():
    """Debug intent analysis flow through the complete workflow path."""
    
    try:
        # Test 1: Create a WorkflowState directly and test intent_classification access
        from composer.graph.state import WorkflowState
        
        # Import the required enums
        from models import ComplexityLevel, ComputationalRequirement
        
        # Create sample intent analysis with engineering fields
        sample_intent = IntentAnalysis(
            workflow_type=WorkflowType.ENGINEERING,
            complexity_level=ComplexityLevel.COMPLEX, 
            required_capabilities=[],
            domain_specificity=0.8,
            reusability_potential=0.6,
            confidence=0.9,
            response_format=ResponseFormat.CODE_SOLUTION,  # This should NOT be None
            technical_domain=TechnicalDomain.SOFTWARE_DEVELOPMENT,
            requires_tools=True,
            requires_custom_tools=False,
            tool_complexity_score=0.7,
            computational_requirements=ComputationalRequirement.MODERATE
        )
        
        print(f"✅ Created sample intent analysis:")
        print(f"   workflow_type: {sample_intent.workflow_type}")
        print(f"   technical_domain: {sample_intent.technical_domain}")
        print(f"   response_format: {sample_intent.response_format}")
        print(f"   response_format type: {type(sample_intent.response_format)}")
        
        # Test 2: Test state updates like the planning_intent subgraph does
        # Skip full WorkflowState creation due to UserConfig complexity
        # Instead, just create a mock object to test setattr behavior
        
        class MockState:
            def __init__(self):
                self.intent_classification = []
                
        mock_state = MockState()
        
        print(f"\n🔧 Initial state intent_classification: {mock_state.intent_classification}")
        
        # Test 3: Simulate planning_intent transform_to_main_state
        updates = {"intent_classification": [sample_intent]}
        
        # Apply updates using setattr like the workflow does
        for key, value in updates.items():
            setattr(mock_state, key, value)
            
        print(f"🔧 After setattr update: {mock_state.intent_classification}")
        print(f"🔧 First intent response_format: {mock_state.intent_classification[0].response_format}")
        
        # Test 4: Simulate engineering node access pattern
        for intent in mock_state.intent_classification:
            domain = intent.technical_domain
            response_format = intent.response_format
            
            print(f"\n🔍 Engineering node simulation:")
            print(f"   intent.technical_domain: {intent.technical_domain}")
            print(f"   intent.response_format: {intent.response_format}")
            print(f"   domain variable: {domain}")
            print(f"   response_format variable: {response_format}")
            
            # Check the exact condition from engineering.py
            if not domain or not response_format:
                print(f"❌ WOULD FAIL: domain={domain}, response_format={response_format}")
                return
            else:
                print(f"✅ WOULD PASS: domain={domain}, response_format={response_format}")
        
        # Test 5: Check JSON serialization/deserialization (might be where the issue is)
        import json
        from pydantic import BaseModel
        
        # Test if the IntentAnalysis survives JSON round-trip
        intent_dict = sample_intent.model_dump()
        print(f"\n📄 JSON serialized: {intent_dict}")
        
        intent_restored = IntentAnalysis.model_validate(intent_dict)
        print(f"📄 JSON restored response_format: {intent_restored.response_format}")
        
        # Test 6: Check if enum handling is consistent
        print(f"\n🏷️ Enum testing:")
        print(f"   Original enum value: {sample_intent.response_format}")
        print(f"   Enum name: {sample_intent.response_format.name if sample_intent.response_format else 'None'}")
        print(f"   Enum value: {sample_intent.response_format.value if sample_intent.response_format else 'None'}")
        
        print(f"\n✅ All workflow intent flow tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Workflow intent flow test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    asyncio.run(debug_workflow_intent_flow())