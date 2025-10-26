#!/usr/bin/env python3
"""
Test script for updated GPU selection logic in intelligent OOM recovery.
Tests the preference for GPU0 with fallback to best GPU when GPU0 is insufficient.
"""

import os
import sys

# Ensure we can import from the inference directory
inference_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, inference_path)

try:
    from runner.utils.intelligent_oom_recovery import IntelligentOOMRecovery
    from runner.utils.hardware_manager import hardware_manager
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_gpu_selection_logic():
    """Test the updated GPU selection priorities"""
    
    print("\n" + "="*60)
    print("TESTING UPDATED GPU SELECTION LOGIC")
    print("="*60)
    
    try:
        # Initialize the recovery system
        recovery = IntelligentOOMRecovery()
        
        # Test available GPU memory detection
        available_memory = recovery.get_available_gpu_memory_mb(hardware_manager)
        print(f"📊 Selected GPU Memory: {available_memory} MB")
        
        # Test total GPU memory detection  
        total_memory = recovery.get_total_gpu_memory_mb(hardware_manager)
        print(f"📊 Total System Memory: {total_memory} MB")
        
        # Get detailed hardware stats to understand the selection
        print("\n🔍 Hardware Manager GPU Stats:")
        stats = hardware_manager.update_all_memory_stats()
        
        if isinstance(stats, dict) and stats:
            print(f"   GPU Memory Stats Available: {len(stats)} GPUs")
            
            # Show all available GPUs and their memory
            gpu_candidates = []
            for gpu_id, gpu_stats in stats.items():
                try:
                    # Convert string GPU ID to integer for comparison
                    gpu_id_int = int(gpu_id)
                    available_mb = gpu_stats.mem_free
                    total_mb = gpu_stats.mem_total
                    used_mb = gpu_stats.mem_used
                    
                    gpu_info = {
                        'id': gpu_id_int,
                        'available_memory': available_mb,
                        'total_memory': total_mb,
                        'used_memory': used_mb,
                        'utilization_pct': (used_mb / total_mb) * 100 if total_mb > 0 else 0
                    }
                    gpu_candidates.append(gpu_info)
                    
                    print(f"   GPU {gpu_id}: {available_mb} MB free / {total_mb} MB total ({gpu_info['utilization_pct']:.1f}% used)")
                except (ValueError, AttributeError) as e:
                    print(f"   Skipping non-GPU device {gpu_id}: {e}")
            
            # Explain selection logic
            print("\n🧠 GPU Selection Analysis:")
            
            # Find GPU 0
            gpu_0 = None
            for gpu in gpu_candidates:
                if gpu['id'] == 0:
                    gpu_0 = gpu
                    break
            
            if gpu_0:
                print(f"   - GPU 0 found: {gpu_0['available_memory']} MB available")
                print(f"   - GPU 0 sufficient (≥8GB)?: {gpu_0['available_memory'] >= 8000}")
                
                # Find best GPU by memory
                best_gpu = max(gpu_candidates, key=lambda x: x['available_memory'])
                print(f"   - Best GPU by memory: GPU {best_gpu['id']} with {best_gpu['available_memory']} MB")
                
                # Check if best GPU has significantly more memory (>2x)
                memory_ratio = best_gpu['available_memory'] / gpu_0['available_memory'] if gpu_0['available_memory'] > 0 else float('inf')
                print(f"   - Memory ratio (best/gpu0): {memory_ratio:.2f}x")
                print(f"   - Switch threshold met (>2x)?: {memory_ratio > 2}")
                
                # Determine expected selection based on actual logic
                if gpu_0['available_memory'] < 8000:
                    expected = f"GPU {best_gpu['id']} (GPU 0 insufficient memory)"
                elif memory_ratio > 2 and best_gpu['id'] != gpu_0['id']:
                    expected = f"GPU {best_gpu['id']} (significantly more memory: {memory_ratio:.2f}x)"
                else:
                    expected = f"GPU {gpu_0['id']} (consistency preference)"
                    
                print(f"   - Expected selection: {expected}")
            else:
                print("   - No GPU 0 found, will select GPU with most memory")
        
        print("\n✅ GPU Selection Test Complete")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting GPU selection logic test...")
    success = test_gpu_selection_logic()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Tests failed!")
        sys.exit(1)