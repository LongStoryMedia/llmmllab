#!/usr/bin/env python3
"""Test bind_tools implementation in BaseLlamaCppPipeline."""

import sys
import traceback

from runner.pipeline_factory import pipeline_factory

def test_bind_tools():
    """Test that bind_tools method works without errors."""
    print("🧪 Testing bind_tools implementation...")
    
    try:
        # Use global pipeline factory
        factory = pipeline_factory
        print(f"✅ Using global factory with {len(factory.models)} models")
        
        # Get a model from the models - use a text-to-text model
        model_name = "llama-chat-summary-3_2-3b-q5-k-m"
        if model_name not in factory.models:
            available_models = list(factory.models.keys())
            print(f"❌ Model {model_name} not found. Available models: {available_models[:5]}...")
            return False
            
        model = factory.models[model_name]
        print(f"✅ Found model: {model.name}")
        
        # Create a basic profile with required fields
        from models import ModelProfile, ModelProfileType
        profile = ModelProfile(
            name="test_profile",
            model_name=model.name,
            parameters={},
            user_id="test_user",
            system_prompt="You are a helpful assistant.",
            type=ModelProfileType.Primary
        )
        
        # Create a pipeline
        pipeline = factory.create_pipeline(model, profile)
        print(f"✅ Successfully created pipeline: {type(pipeline).__name__}")
        
        # Test bind_tools method
        print("📎 Testing bind_tools method...")
        
        # Create some mock tools
        mock_tools = [
            {"type": "function", "function": {"name": "test_tool", "description": "A test tool"}},
            {"type": "function", "function": {"name": "another_tool", "description": "Another test tool"}}
        ]
        
        # Call bind_tools
        bound_pipeline = pipeline.bind_tools(mock_tools)
        print(f"✅ Successfully bound tools: {type(bound_pipeline).__name__}")
        
        # Verify the bound tools are stored
        if hasattr(bound_pipeline, '_bound_tools'):
            print(f"✅ Bound tools stored: {len(bound_pipeline._bound_tools)} tools")
        else:
            print("❌ Bound tools not found on pipeline")
            return False
            
        # Verify it's a new instance
        if bound_pipeline is not pipeline:
            print("✅ bind_tools returned new instance (correct)")
        else:
            print("❌ bind_tools returned same instance (incorrect)")
            return False
            
        print("🎉 All bind_tools tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("Traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bind_tools()
    sys.exit(0 if success else 1)