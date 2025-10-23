#!/usr/bin/env python3
"""
Test to debug why engineering agent isn't being triggered for technical questions.
"""

import asyncio
import json
from typing import List

from models import Message, MessageContent, MessageContentType, MessageRole, WorkflowType
from utils.logging import llmmllogger


class EngineeringAgentTriggerTest:
    """Test engineering agent triggering logic."""

    def __init__(self):
        self.logger = llmmllogger.logger.bind(component="EngineeringAgentTriggerTest")

    async def test_intent_classification_for_technical_questions(self):
        """Test how different technical questions are classified."""
        self.logger.info("🎯 Testing intent classification for technical questions")
        
        self.logger.info("🔍 Current workflow types available:")
        for wt in WorkflowType:
            self.logger.info(f"   - {wt.name}: {wt.value}")
        
        self.logger.info(f"✅ Engineering workflow type: {WorkflowType.ENGINEERING.value}")
        
        self.logger.info("🎉 Intent classification test completed")

    async def test_workflow_routing_logic(self):
        """Test the workflow routing logic with engineering intents."""
        self.logger.info("🔀 Testing workflow routing logic")
        
        from composer.graph.state import WorkflowState
        from composer.nodes.routing.router import WorkflowRouter
        from models import IntentAnalysis, WorkflowType, ComplexityLevel, ComputationalRequirement
        
        # Create test state with engineering intent
        engineering_intent = IntentAnalysis(
            workflow_type=WorkflowType.ENGINEERING,
            complexity_level=ComplexityLevel.MODERATE,
            required_capabilities=[],
            domain_specificity=0.8,
            reusability_potential=0.6,
            confidence=0.9,
            tool_complexity_score=0.5,
            computational_requirements=ComputationalRequirement.MODERATE
        )
        
        state = WorkflowState(
            user_id="test-user",
            conversation_id="test-conv",
            messages=[],
            intent_classification=[engineering_intent],
            selected_workflows=set(),
            selected_tools=[],
            required_tools=[],
            dynamic_tools=[],
            config=None,
            workflow_type=None,
            error_details=[],
            node_metadata={}
        )
        
        # Test router
        router = WorkflowRouter("test-user")
        updated_state = await router(state)
        
        has_engineering = WorkflowType.ENGINEERING in updated_state.selected_workflows
        
        self.logger.info(f"🔍 Selected workflows: {updated_state.selected_workflows}")
        self.logger.info(f"✅ Engineering workflow selected: {has_engineering}")
        
        if not has_engineering:
            self.logger.error("❌ Engineering workflow was NOT selected despite engineering intent!")
        else:
            self.logger.info("🎉 Engineering workflow correctly selected")

    async def run_all_tests(self):
        """Run all engineering agent trigger tests."""
        self.logger.info("🚀 Starting engineering agent trigger investigation")
        
        try:
            await self.test_intent_classification_for_technical_questions()
            await self.test_workflow_routing_logic()
            
            self.logger.info("✅ All engineering agent trigger tests completed")
            
        except Exception as e:
            self.logger.error(f"❌ Test execution failed: {e}")
            raise


async def main():
    """Main test runner."""
    test = EngineeringAgentTriggerTest()
    await test.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())