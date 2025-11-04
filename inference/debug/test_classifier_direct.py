#!/usr/bin/env python3
"""
Direct test of classifier agent with enhanced prompting.
"""
import asyncio
import os
import sys

# Add the inference directory to the path for imports
sys.path.insert(0, '/app')

from composer.agents.classifier_agent import ClassifierAgent
from models import (
    Message, MessageContent, MessageContentType, MessageRole,
    ModelProfile, NodeMetadata, PipelinePriority
)
from utils.model_profile import get_model_profile
from runner import PipelineFactory

async def test_classifier_direct():
    """Test classifier with enhanced constraints."""
    print("🔍 Testing ClassifierAgent with enhanced constraints...")
    
    try:
        # Use global pipeline factory instance
        from runner import pipeline_factory
        
        # Get classifier model profile using default profiles
        from models import default_model_profiles, ModelTask
        profile = default_model_profiles.DEFAULT_PRIMARY_PROFILE
        print(f"✅ Got model profile: {profile.model_name}")
        
        # Create node metadata
        metadata = NodeMetadata(
            node_id='test_classifier_001',
            node_name='TestClassifier', 
            node_type='ClassifierTest',
            user_id='test_user'
        )
        
        # Create classifier agent
        classifier = ClassifierAgent(pipeline_factory, profile, metadata)
        print("✅ Created ClassifierAgent")
        
        # Create test message
        test_message = Message(
            content=[MessageContent(
                text="Look at this image and describe what you see. What colors are visible, and what might this represent? Also, please provide information about the latest developments in multimodal AI models.",
                type=MessageContentType.TEXT
            )],
            role=MessageRole.USER
        )
        
        # Run classification
        print("🎯 Running intent analysis...")
        results = await classifier.analyze([test_message], [])
        
        print(f"✅ Classification completed with {len(results)} intents")
        
        # Check results
        for i, intent in enumerate(results):
            print(f"\nIntent {i+1}:")
            print(f"  - tool_complexity_score: {intent.tool_complexity_score} (type: {type(intent.tool_complexity_score)})")
            print(f"  - domain_specificity: {intent.domain_specificity}")
            print(f"  - confidence: {intent.confidence}")
            print(f"  - workflow_type: {intent.workflow_type}")
            
            # Validate scores
            valid = True
            if not (0.0 <= intent.tool_complexity_score <= 1.0):
                print(f"  ❌ INVALID tool_complexity_score: {intent.tool_complexity_score}")
                valid = False
            if not (0.0 <= intent.domain_specificity <= 1.0):
                print(f"  ❌ INVALID domain_specificity: {intent.domain_specificity}")
                valid = False
            if not (0.0 <= intent.confidence <= 1.0):
                print(f"  ❌ INVALID confidence: {intent.confidence}")
                valid = False
                
            if valid:
                print("  ✅ All scores within valid range")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_classifier_direct())
    sys.exit(0 if success else 1)