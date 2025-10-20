#!/usr/bin/env python3
"""
Test script to debug LLaMA initialization failures
"""

import sys
import traceback
from uuid import UUID
import datetime

# Add to path for imports
sys.path.append('/app')

from runner.pipelines.txt2txt.qwen3moe import Qwen3Moe
from models.model import Model, ModelProvider, ModelTask, ModelDetails
from models.model_profile import ModelProfile, ModelParameters

def test_llama_initialization():
    """Test LLaMA initialization with different context sizes"""
    
    # Create test model
    model = Model(
        id='qwen3-4b-ud-q6-k-xl',
        name='Qwen3-4B',
        model='/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf',
        task=ModelTask.TEXTTOTEXT,
        modified_at='2025-07-20',
        size=3658223392,
        digest='qwen3-4b-ud-q6-k-xl-20250720',
        details=ModelDetails(
            parent_model='Qwen/Qwen3-4B',
            format='gguf',
            gguf_file='/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf',
            family='qwen',
            families=['Qwen', 'MoE'],
            parameter_size='4B',
            quantization_level='Q6_K_XL',
            dtype='BF16',
            precision='fp16',
            specialization='TextToText',
            description=None,
            weight=1.0
        ),
        pipeline='Qwen3Pipe',
        lora_weights=[],
        provider=ModelProvider.LLAMA_CPP
    )

    # Test different context sizes
    contexts_to_test = [4096, 8192, 16384, 32768, 40960]
    
    for ctx_size in contexts_to_test:
        print(f"\n🧪 Testing context size: {ctx_size}")
        
        try:
            # Create test profile
            profile = ModelProfile(
                id=UUID('00000000-0000-0000-0000-000000000009'),
                user_id='system',
                name='Analysis (Default)',
                description='Profile for detailed analysis of text.',
                model_name='qwen3-4b-ud-q6-k-xl',
                parameters=ModelParameters(
                    num_ctx=ctx_size,
                    repeat_last_n=-1,
                    repeat_penalty=1.05,
                    temperature=0.7,
                    seed=0,
                    stop=['<|im_end|>', '<|endoftext|>', '<|end|>'],
                    num_predict=-1,
                    top_k=20,
                    top_p=0.8,
                    min_p=0.0,
                    think=False,
                    max_tokens=16384,
                    n_parts=-1,
                    batch_size=None,
                    n_cpu_moe=0,
                    reasoning_effort='medium',
                    flash_attention=True
                ),
                system_prompt='Test prompt',
                created_at=datetime.datetime.now(),
                updated_at=datetime.datetime.now(),
                model_version=None,
                type=9,
                image_settings=None,
                circuit_breaker=None,
                gpu_config=None
            )
            
            print(f"   🚀 Initializing pipeline...")
            pipeline = Qwen3Moe(model=model, profile=profile)
            print(f"   ✅ SUCCESS: Context {ctx_size} initialized")
            
            # Clean up
            try:
                pipeline.close()
                print(f"   🧹 Pipeline closed cleanly")
            except Exception as cleanup_error:
                print(f"   ⚠️  Cleanup error: {cleanup_error}")
                
        except Exception as e:
            print(f"   ❌ FAILED: Context {ctx_size} - {e}")
            print(f"   📋 Full error:")
            traceback.print_exc()
            
        print(f"   " + "="*50)

if __name__ == "__main__":
    print("🔬 Testing LLaMA Initialization with Different Context Sizes")
    print("="*60)
    test_llama_initialization()
    print("\n✅ Test completed")