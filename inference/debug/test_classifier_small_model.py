#!/usr/bin/env python3
"""
Test classifier agent with smaller model (qwen3-4b-ud-q6-k-xl) to verify auto-correction.
"""
import asyncio
import os
import sys

# Add the inference directory to the path for imports
sys.path.insert(0, '/app')

from composer.agents.classifier_agent import ClassifierAgent
from models import (
    Message, MessageContent, MessageContentType, MessageRole,
    ModelProfile, NodeMetadata, PipelinePriority, default_model_profiles
)
from runner import pipeline_factory


async def test_classifier_small_model():
    """Test classifier with smaller model to verify auto-correction."""
    print("🔍 Testing ClassifierAgent with smaller model for auto-correction...")
    
    try:
        # Use analysis profile (uses qwen3-4b-ud-q6-k-xl) which tends to generate invalid values
        profile = default_model_profiles.DEFAULT_ANALYSIS_PROFILE
        print(f"✅ Using analysis profile with model: {profile.model_name}")
        
        # Create node metadata
        metadata = NodeMetadata(
            node_id='test_classifier_small_001',
            node_name='TestClassifierSmall', 
            node_type='ClassifierSmallTest',
            user_id='test_user'
        )
        
        # Create classifier agent
        classifier = ClassifierAgent(pipeline_factory, profile, metadata)
        print("✅ Created ClassifierAgent with smaller model")
        
        # Create test message that often triggers tool_complexity_score > 1
        test_message = Message(
            content=[MessageContent(
                type=MessageContentType.TEXT,
                text="I need to build a complex machine learning pipeline with multi-modal data processing, real-time inference, distributed training across multiple GPUs, and custom CUDA kernels for optimization."
            )],
            role=MessageRole.USER,
            id=1001,
            conversation_id=2001,
            created_at="2025-11-04T05:00:00Z"
        )
        
        print("🎯 Running intent analysis with complex ML pipeline request...")
        
        # Analyze the message (no static tools available for this test)
        intents = await classifier.analyze([test_message], [])
        
        print(f"✅ Analysis completed successfully with {len(intents)} intents")
        
        # Validate all tool_complexity_score values are within range
        valid_scores = True
        for i, intent in enumerate(intents):
            score = intent.tool_complexity_score
            print(f"Intent {i+1}:")
            print(f"  - tool_complexity_score: {score} (type: {type(score)})")
            print(f"  - domain_specificity: {intent.domain_specificity}")
            print(f"  - confidence: {intent.confidence}")
            print(f"  - workflow_type: {intent.workflow_type}")
            
            if not (0.0 <= score <= 1.0):
                print(f"❌ Invalid tool_complexity_score: {score}")
                valid_scores = False
            else:
                print(f"  ✅ Valid score within range")
        
        if valid_scores:
            print("\n🎉 SUCCESS: All tool_complexity_score values are within valid range!")
            print("🔧 Auto-correction is working properly!")
            return True
        else:
            print("\n❌ FAILURE: Some tool_complexity_score values are outside valid range")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_classifier_small_model())
    sys.exit(0 if success else 1)