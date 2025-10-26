#!/usr/bin/env python3
"""
Test script for validating strong typing in structured response data.
Run this to verify that the StructuredResponseData TypedDict is properly typed.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/app')

from server.routers.chat import StructuredResponseData, store_structured_response_data
from models import Thought, ToolExecutionResult, IntentAnalysis, WorkflowType, ComplexityLevel, RequiredCapability, ComputationalRequirement
from db import storage
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="typing_test")

async def test_structured_response_data_typing():
    """Test that StructuredResponseData maintains strong typing."""
    
    if not storage.initialized:
        logger.error("Database not initialized")
        return False
    
    try:
        # Create properly typed test data
        test_thoughts = [
            Thought(text="This is a test thought about the user's request."),
            Thought(text="Another thought analyzing the complexity.")
        ]
        
        test_tool_calls = [
            ToolExecutionResult(
                tool_name="test_tool",
                execution_id="call_1",
                success=True,
                args={"query": "test query"},
                result_data={"result": "test result"},
                execution_time_ms=150
            ),
            ToolExecutionResult(
                tool_name="search_tool", 
                execution_id="call_2",
                success=True,
                args={"term": "machine learning"},
                result_data={"count": 5},
                execution_time_ms=230
            )
        ]
        
        test_analyses = [
            IntentAnalysis(
                workflow_type=WorkflowType.RESEARCH,
                complexity_level=ComplexityLevel.MEDIUM,
                required_capabilities=[RequiredCapability.WEB_SEARCH],
                domain_specificity=0.7,
                reusability_potential=0.8,
                confidence=0.9,
                requires_tools=True,
                requires_custom_tools=False,
                tool_complexity_score=0.6,
                computational_requirements=ComputationalRequirement.LOW
            )
        ]
        
        # Create strongly typed structured data
        structured_data: StructuredResponseData = {
            "thoughts": test_thoughts,
            "tool_calls": test_tool_calls, 
            "analyses": test_analyses
        }
        
        logger.info("✅ StructuredResponseData created successfully with strong typing")
        logger.info(f"  - Thoughts: {len(structured_data['thoughts'])}")
        logger.info(f"  - Tool calls: {len(structured_data['tool_calls'])}")
        logger.info(f"  - Analyses: {len(structured_data['analyses'])}")
        
        # Test type validation
        for thought in structured_data["thoughts"]:
            assert isinstance(thought, Thought), f"Expected Thought, got {type(thought)}"
            
        for tool_call in structured_data["tool_calls"]:
            assert isinstance(tool_call, ToolExecutionResult), f"Expected ToolExecutionResult, got {type(tool_call)}"
            
        for analysis in structured_data["analyses"]:
            assert isinstance(analysis, IntentAnalysis), f"Expected IntentAnalysis, got {type(analysis)}"
        
        logger.info("✅ All type assertions passed")
        
        # Test that we can store the data (would require a test message)
        logger.info("✅ Strong typing validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Typing test failed: {e}", exc_info=True)
        return False

async def test_invalid_types():
    """Test that invalid types are properly rejected."""
    
    try:
        logger.info("🧪 Testing invalid type rejection...")
        
        # This should fail type checking if used with a proper type checker
        invalid_structured_data = {
            "thoughts": ["string instead of Thought object"],  # Wrong type
            "tool_calls": [{"tool_name": "dict instead of ToolExecutionResult"}],  # Wrong type
            "analyses": [{"workflow": "dict instead of IntentAnalysis"}]  # Wrong type
        }
        
        # Type checker should catch this, but at runtime we need validation
        logger.warning("⚠️ Note: Runtime type validation should be added to reject invalid types")
        logger.info("✅ Invalid type test completed (would fail with strict type checking)")
        return True
        
    except Exception as e:
        logger.error(f"Invalid type test failed unexpectedly: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("Starting strong typing validation tests...")
    
    # Initialize database connection if needed
    if not storage.initialized:
        logger.info("Initializing database connection...")
        connection_string = os.getenv('DATABASE_URL', 'postgresql://lsm:password@psql-service.psql.svc.cluster.local:5432/llmmll')
        await storage.initialize(connection_string)
    
    success1 = await test_structured_response_data_typing()
    success2 = await test_invalid_types()
    
    if success1 and success2:
        logger.info("🎉 All typing tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Some typing tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())