#!/usr/bin/env python3
"""Test script to validate Qwen3VL pipeline creation."""

from models import ModelProfile, ModelParameters
from runner.pipeline_factory import pipeline_factory

def test_qwen3_vl_pipeline_creation():
    """Test creating Qwen3VL pipeline instance."""
    print("🚀 Testing Qwen3VL pipeline creation...")
    
    # Create a minimal test profile
    profile = ModelProfile(
        user_id="test-user",
        name="test-qwen3-vl",
        model_name="qwen3-vl-32b-thinking-abliterated",
        system_prompt="You are a helpful AI assistant.",
        type=1,  # Integer type as required by schema
        parameters=ModelParameters(
            num_ctx=4096,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
        ),
    )
    
    try:
        # Test pipeline creation
        pipeline = pipeline_factory.get_pipeline(profile)
        print(f"✅ Pipeline created successfully: {type(pipeline).__name__}")
        print(f"   Model type: {pipeline._llm_type}")
        print(f"   Identifying params: {pipeline._identifying_params}")
        
        # Test mmproj detection
        if hasattr(pipeline, '_get_mmproj_path'):
            mmproj_path = pipeline._get_mmproj_path()
            print(f"   MMPROJ path: {mmproj_path}")
            if mmproj_path:
                print("✅ MMPROJ file detected")
            else:
                print("⚠️  MMPROJ file not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_qwen3_vl_pipeline_creation()
    exit(0 if success else 1)