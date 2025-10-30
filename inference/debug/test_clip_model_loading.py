#!/usr/bin/env python3
"""Debug script to test clip_model_path loading."""

import json
from runner.pipeline_factory import PipelineFactory

def test_clip_model_loading():
    """Test if clip_model_path is properly loaded from .models.json."""
    
    # Load raw JSON data
    with open('/app/.models.json') as f:
        data = json.load(f)
    
    # Find our model
    qwen_data = None
    for model_data in data:
        if model_data['id'] == 'qwen3-vl-32b-thinking-abliterated':
            qwen_data = model_data
            break
    
    if not qwen_data:
        print("❌ Model not found in .models.json")
        return False
    
    print("📄 Raw model data details:")
    print(json.dumps(qwen_data['details'], indent=2))
    
    # Test model creation
    factory = PipelineFactory({})
    model = factory._create_model_from_data(qwen_data)
    
    if not model:
        print("❌ Failed to parse model data")
        return False
    
    print("\n🏗️ Parsed model details:")
    print(f"   clip_model_path: {getattr(model.details, 'clip_model_path', 'NOT FOUND')}")
    print(f"   gguf_file: {getattr(model.details, 'gguf_file', 'NOT FOUND')}")
    print(f"   All fields: {model.details.__dict__}")
    
    return getattr(model.details, 'clip_model_path', None) is not None

if __name__ == "__main__":
    success = test_clip_model_loading()
    print(f"\n✅ Test {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)