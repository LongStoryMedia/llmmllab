#!/usr/bin/env python3
"""Debug memory calculation for Qwen3-4B model."""

import json

def debug_memory_calculation():
    """Debug the memory calculation issue."""
    print("🔍 Debugging memory calculation for Qwen3-4B...")
    
    # Load the model configuration
    with open('/app/.models.json', 'r') as f:
        models_data = json.load(f)
    
    qwen_model = None
    for model in models_data:
        if model.get('name') == 'Qwen3-4B':
            qwen_model = model
            break
    
    if not qwen_model:
        print("❌ Qwen3-4B model not found")
        return
    
    print(f"📊 Model configuration:")
    print(f"  Name: {qwen_model.get('name')}")
    print(f"  Size: {qwen_model.get('size')} bytes ({qwen_model.get('size') / 1e9:.2f}GB)")
    
    details = qwen_model.get('details', {})
    print(f"  Parameter size: {details.get('parameter_size')}")
    print(f"  Quantization: {details.get('quantization_level')}")
    
    # Simulate the memory calculation logic
    base = 512 * 1024 * 1024  # 512MB
    model_size = 0
    
    # Try to parse parameter size
    if details.get('parameter_size'):
        try:
            raw = details['parameter_size'].upper().strip()
            print(f"  Raw parameter size: {raw}")
            
            if raw.endswith("B"):
                params = float(raw[:-1]) * 1_000_000_000
                print(f"  Parsed parameters: {params:,.0f}")
                
                # Quantization calculation
                q = (details.get('quantization_level') or 'q4').lower()
                print(f"  Quantization level: {q}")
                
                if "q6" in q:
                    bpp = 0.75  # 6-bit quantization should be ~0.75 bytes per parameter
                elif "q4" in q or "iq4" in q:
                    bpp = 0.5
                elif "q5" in q:
                    bpp = 0.625
                elif "q8" in q:
                    bpp = 1.0
                elif any(x in q for x in ["fp16", "bf16", "f16"]):
                    bpp = 2.0
                else:
                    bpp = 4.0
                
                print(f"  Bits per parameter: {bpp}")
                model_size = int(params * bpp)
                print(f"  Calculated model size: {model_size:,} bytes ({model_size / 1e9:.2f}GB)")
                
        except Exception as e:
            print(f"  Error parsing parameter size: {e}")
    
    # Fallback to file size
    if model_size == 0:
        file_size = qwen_model.get('size', 0)
        if file_size and file_size < 100 * 1024 * 1024 * 1024:  # Less than 100GB
            model_size = file_size
            print(f"  Using file size: {model_size:,} bytes ({model_size / 1e9:.2f}GB)")
    
    # Default fallback for TextToText
    if model_size == 0:
        model_size = 4 * 1024 * 1024 * 1024  # 4GB default
        print(f"  Using default size: {model_size:,} bytes ({model_size / 1e9:.2f}GB)")
    
    # Context memory (10% of model size, max 2GB)
    context_mem = min(model_size * 0.1, 2 * 1024 * 1024 * 1024)
    print(f"  Context memory: {context_mem:,} bytes ({context_mem / 1e9:.2f}GB)")
    
    # Total with safety buffer
    total = (base + model_size + context_mem) * 1.10
    print(f"  Base memory: {base:,} bytes ({base / 1e9:.2f}GB)")
    print(f"  Total with 1.1x buffer: {total:,} bytes ({total / 1e9:.2f}GB)")
    
    print("✅ Memory calculation debug completed")

if __name__ == "__main__":
    debug_memory_calculation()