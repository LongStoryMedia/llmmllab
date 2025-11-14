#!/usr/bin/env python3
"""
Simple Large Model Memory Collection

Runs directly inside the container to collect memory samples from large models
without needing kubectl commands. Focuses on conservative configurations.
"""

import subprocess
import json
import time
import logging
import os
import signal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimpleTestConfig:
    """Simple test configuration for in-container execution."""
    model_id: str
    model_name: str
    param_size: str
    gguf_path: str
    context_size: int
    batch_size: int
    ubatch_size: int
    gpu_layers: int
    mmproj_path: Optional[str] = None
    notes: str = ""

def get_gpu_memory_usage() -> Tuple[bool, Dict[str, float]]:
    """Get current GPU memory usage."""
    try:
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            logger.error(f"nvidia-smi failed: {result.stderr}")
            return False, {}
        
        memory_values = result.stdout.strip().split('\n')
        memory_dict = {}
        
        for i, value in enumerate(memory_values):
            try:
                memory_dict[f"gpu{i}"] = float(value.strip())
            except ValueError:
                logger.warning(f"Invalid memory value for GPU {i}: {value}")
                memory_dict[f"gpu{i}"] = -1.0
        
        return True, memory_dict
        
    except Exception as e:
        logger.error(f"Error getting GPU memory: {e}")
        return False, {}

def estimate_memory_conservative(config: SimpleTestConfig) -> float:
    """Conservative memory estimation."""
    base_sizes = {
        "32B": 18.0,  # Conservative 18GB for 32B models
        "30B": 16.0,  # Conservative 16GB for 30B models
    }
    
    base_gb = base_sizes.get(config.param_size, 10.0)
    context_overhead = (config.context_size / 4096) * 0.5  # 0.5GB per 4K context
    clip_overhead = 1.2 if config.mmproj_path else 0.0
    safety_buffer = 2.0
    
    total = base_gb + context_overhead + clip_overhead + safety_buffer
    return total

def run_simple_test(config: SimpleTestConfig) -> Optional[Dict]:
    """Run a single test using direct llama-server execution."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {config.model_name}")
    logger.info(f"Context: {config.context_size//1024}K, Batch: {config.batch_size}")
    logger.info(f"{'='*60}")
    
    # Check if model file exists
    if not os.path.exists(config.gguf_path):
        logger.error(f"Model file not found: {config.gguf_path}")
        return None
    
    if config.mmproj_path and not os.path.exists(config.mmproj_path):
        logger.error(f"MMProj file not found: {config.mmproj_path}")
        return None
    
    # Estimate memory
    estimated_gb = estimate_memory_conservative(config)
    if estimated_gb > 28:  # Conservative limit
        logger.warning(f"Estimated memory {estimated_gb:.1f}GB too high, skipping")
        return None
    
    # Get baseline memory
    baseline_success, baseline_memory = get_gpu_memory_usage()
    if not baseline_success:
        logger.error("Failed to get baseline memory")
        return None
    
    baseline_total = sum(baseline_memory.values())
    logger.info(f"Baseline GPU memory: {baseline_total:.0f}MB")
    
    try:
        # Build llama-server command
        cmd = [
            "/llama.cpp/build/bin/llama-server",
            "--model", config.gguf_path,
            "--ctx-size", str(config.context_size),
            "--batch-size", str(config.batch_size),
            "--ubatch-size", str(config.ubatch_size),
            "--gpu-layers", str(config.gpu_layers),
            "--threads", "16",
            "--no-kv-offload",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--numa", "distribute",
            "--split-mode", "layer",
            "--tensor-split", "0.25,0.45,0.30",
            "--main-gpu", "1",
            "--port", "8080",
            "--timeout", "30"
        ]
        
        if config.mmproj_path:
            cmd.extend(["--mmproj", config.mmproj_path])
        
        logger.info("Starting llama-server...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Start server
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Wait for startup
        startup_time = 0
        max_startup = 45
        
        while startup_time < max_startup:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server crashed during startup:")
                logger.error(f"STDOUT: {stdout[-500:] if stdout else 'None'}")
                logger.error(f"STDERR: {stderr[-500:] if stderr else 'None'}")
                return None
            
            time.sleep(1)
            startup_time += 1
            
            if startup_time % 10 == 0:
                logger.info(f"Startup progress: {startup_time}/{max_startup}s")
        
        # Give it time to fully load
        logger.info("Allowing time for model loading...")
        time.sleep(10)
        
        # Check if still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"Server died after startup:")
            logger.error(f"STDOUT: {stdout[-500:] if stdout else 'None'}")
            logger.error(f"STDERR: {stderr[-500:] if stderr else 'None'}")
            return None
        
        # Get memory usage
        success, memory_usage = get_gpu_memory_usage()
        if not success:
            logger.error("Failed to get memory usage")
            return None
        
        logger.info("Memory measurement successful!")
        for gpu, usage in memory_usage.items():
            logger.info(f"  {gpu.upper()}: {usage:.0f}MB")
        
        total_actual_mb = sum(memory_usage.values())
        total_actual_gb = total_actual_mb / 1024
        
        logger.info(f"Total GPU memory: {total_actual_gb:.2f}GB")
        
        accuracy_ratio = estimated_gb / total_actual_gb if total_actual_gb > 0 else 0
        
        result = {
            "model_id": config.model_id,
            "model_name": config.model_name,
            "param_size": config.param_size,
            "gguf_path": config.gguf_path,
            "context_size": config.context_size,
            "batch_size": config.batch_size,
            "ubatch_size": config.ubatch_size,
            "gpu_layers": config.gpu_layers,
            "mmproj_path": config.mmproj_path,
            "estimated_gb": estimated_gb,
            "actual_gpu0_mb": memory_usage.get("gpu0", -1),
            "actual_gpu1_mb": memory_usage.get("gpu1", -1),
            "actual_gpu2_mb": memory_usage.get("gpu2", -1),
            "total_actual_gb": total_actual_gb,
            "accuracy_ratio": accuracy_ratio,
            "notes": config.notes,
            "simple_container_test": True,
            "production_config": {
                "threads": 16,
                "no_kv_offload": True,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "numa": "distribute",
                "split_mode": "layer",
                "tensor_split": "0.25,0.45,0.30",
                "main_gpu": 1,
                "timeout": 30
            }
        }
        
        logger.info(f"✅ Success: {config.model_name} @ {config.context_size//1024}K")
        logger.info(f"   Estimated: {estimated_gb:.1f}GB, Actual: {total_actual_gb:.1f}GB ({accuracy_ratio:.2f}x)")
        
        return result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None
        
    finally:
        # Always try to stop the server
        try:
            if 'process' in locals() and process.poll() is None:
                logger.info("Stopping llama-server...")
                # Kill entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown failed, force killing...")
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.wait()
        except Exception as e:
            logger.warning(f"Error stopping server: {e}")
        
        # Wait before next test
        logger.info("Cooling down...")
        time.sleep(10)

def main():
    """Run simple large model memory collection."""
    
    logger.info("Starting Simple Large Model Memory Collection")
    logger.info("Running directly inside container")
    
    # Simple test configurations - conservative settings
    test_configs = [
        # Start with the most conservative tests
        SimpleTestConfig(
            model_id="qwen3-30b-a3b",
            model_name="Qwen3-30B-A3B",
            param_size="30B",
            gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
            context_size=32768,   # 32K - very conservative
            batch_size=1024,      # Small batch
            ubatch_size=1024,
            gpu_layers=-1,
            mmproj_path=None,
            notes="30B MoE conservative 32K"
        ),
        
        SimpleTestConfig(
            model_id="qwen3-30b-a3b",
            model_name="Qwen3-30B-A3B", 
            param_size="30B",
            gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
            context_size=65536,   # 64K
            batch_size=2048,      # Medium batch
            ubatch_size=2048,
            gpu_layers=-1,
            mmproj_path=None,
            notes="30B MoE medium 64K"
        ),
        
        SimpleTestConfig(
            model_id="qwen3-coder-30b-a3b",
            model_name="Qwen3-Coder-30B-A3B",
            param_size="30B",
            gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
            context_size=32768,   # 32K - conservative  
            batch_size=1024,      # Small batch
            ubatch_size=1024,
            gpu_layers=-1,
            mmproj_path=None,
            notes="30B Coder conservative 32K"
        ),
    ]
    
    results = []
    successful_tests = 0
    failed_tests = 0
    
    logger.info(f"Planning to test {len(test_configs)} configurations")
    
    for i, config in enumerate(test_configs, 1):
        logger.info(f"\n--- Test {i}/{len(test_configs)} ---")
        
        result = run_simple_test(config)
        
        if result:
            results.append(result)
            successful_tests += 1
            logger.info(f"✅ Test {i} successful")
        else:
            failed_tests += 1
            logger.error(f"❌ Test {i} failed")
            
            # Wait longer after failure
            logger.info("Extended cooling down after failure...")
            time.sleep(20)
    
    # Save results
    if results:
        output_file = "simple_large_model_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"SIMPLE COLLECTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Successful tests: {successful_tests}")
        logger.info(f"Failed tests: {failed_tests}")
        logger.info(f"Results saved to: {output_file}")
        
        logger.info(f"\nSuccessful measurements:")
        for result in results:
            logger.info(f"  {result['model_name']} @ {result['context_size']//1024}K: "
                       f"{result['total_actual_gb']:.1f}GB actual "
                       f"(est: {result['estimated_gb']:.1f}GB, {result['accuracy_ratio']:.2f}x)")
    else:
        logger.error("No successful tests completed!")

if __name__ == "__main__":
    main()