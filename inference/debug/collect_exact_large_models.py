#!/usr/bin/env python3
"""
Exact Large Model Collection Script

Collects memory samples from the specific models requested:
- qwen3-vl-32b-thinking
- qwen3-30b-a3b  
- qwen3-coder-30b-a3b
- qwen3-vl-30b-a3b-thinking

With production-like configurations matching actual usage patterns.
"""

import subprocess
import json
import time
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ExactTestConfiguration:
    """Exact test configuration for the requested models"""
    
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
    # Production flags
    threads: int = 24
    no_kv_offload: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    numa: str = "distribute"
    n_cpu_moe: int = 5
    split_mode: str = "layer"
    tensor_split: str = "0.22,0.5,0.28"
    main_gpu: int = 1
    no_warmup: bool = True
    cont_batching: bool = True
    no_webui: bool = True
    metrics: bool = True
    jinja: bool = True


@dataclass 
class ExactMemoryResult:
    """Memory measurement result"""
    
    config: ExactTestConfiguration
    estimated_gb: float
    actual_gpu0_mb: float
    actual_gpu1_mb: float
    actual_gpu2_mb: float
    total_actual_gb: float
    accuracy_ratio: float
    success: bool
    error_msg: str = ""
    oom_detected: bool = False
    exceeds_48gb_vram: bool = False


class ExactModelSampleCollector:
    """Collects memory samples from the exact requested models"""
    
    def __init__(self, k8s_pod_name: str):
        self.pod_name = k8s_pod_name
        self.namespace = "ollama"
        self.results: List[ExactMemoryResult] = []
        
    def _simple_memory_estimate(self, config: ExactTestConfiguration) -> float:
        """Simple memory estimate based on model size and context"""
        try:
            # Check model file size
            model_size_gb = 0
            if "32b" in config.model_name.lower():
                model_size_gb = 20  # ~20GB for Q4_K_M 32B model
            elif "30b" in config.model_name.lower():
                model_size_gb = 18  # ~18GB for Q4_K_M 30B model
            else:
                model_size_gb = 15  # Fallback
                
            # Context overhead (rough estimate)
            ctx_gb = config.context_size * 4 * 2 / (1024**3)  # 4 bytes per token, 2 for k+v cache
            
            # Batch overhead
            batch_gb = config.batch_size * 4 / (1024**3)
            
            # MMPROJ overhead
            mmproj_gb = 1.0 if config.mmproj_path else 0.0
            
            total_estimate = model_size_gb + ctx_gb + batch_gb + mmproj_gb + 2.0  # 2GB overhead
            
            return total_estimate
            
        except Exception as e:
            print(f"❌ Simple estimation error: {e}")
            return 25.0  # Conservative fallback
            
    def _get_baseline_memory(self) -> Tuple[float, float, float]:
        """Get baseline GPU memory usage"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace, self.pod_name, "--",
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10, check=True)
            
            memory_values = [float(x.strip()) for x in result.stdout.strip().split('\n')]
            while len(memory_values) < 3:
                memory_values.append(0.0)
                
            return tuple(memory_values[:3])
            
        except Exception as e:
            print(f"❌ Failed to get baseline memory: {e}")
            return (4.0, 4.0, 4.0)
            
    def _get_actual_memory(self) -> Tuple[float, float, float]:
        """Get actual GPU memory usage"""
        try:
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace, self.pod_name, "--",
                "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10, check=True)
            
            memory_values = [float(x.strip()) for x in result.stdout.strip().split('\n')]
            while len(memory_values) < 3:
                memory_values.append(0.0)
                
            return tuple(memory_values[:3])
            
        except Exception as e:
            print(f"❌ Failed to get actual memory: {e}")
            return (0.0, 0.0, 0.0)
            
    def _cleanup_processes(self):
        """Clean up any remaining llama-server processes"""
        try:
            subprocess.run([
                "kubectl", "exec", "-n", self.namespace, self.pod_name, "--",
                "pkill", "-f", "llama-server"
            ], capture_output=True, timeout=10, check=False)
            
            time.sleep(3)
            
            # Verify cleanup with pgrep
            result = subprocess.run([
                "kubectl", "exec", "-n", self.namespace, self.pod_name, "--",
                "pgrep", "-f", "llama-server"
            ], capture_output=True, text=True, timeout=10, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"🧹 Found remaining processes: {result.stdout.strip()}")
                # Force kill
                subprocess.run([
                    "kubectl", "exec", "-n", self.namespace, self.pod_name, "--",
                    "pkill", "-9", "-f", "llama-server"
                ], capture_output=True, timeout=10, check=False)
                time.sleep(2)
            else:
                print("🧹 No additional llama-server processes found")
                
        except Exception as e:
            print(f"🧹 Cleanup warning: {e}")
            
    def _run_exact_test(self, config: ExactTestConfiguration) -> ExactMemoryResult:
        """Run exact memory test for specified model"""
        
        # Format context size
        if config.context_size >= 1048576:
            ctx_display = f"{config.context_size//1048576}M"
        elif config.context_size >= 1024:
            ctx_display = f"{config.context_size//1024}K"
        else:
            ctx_display = f"{config.context_size}"
            
        print(f"\n🧪 Testing {config.model_name} @ {ctx_display} ctx, batch={config.batch_size}, ubatch={config.ubatch_size}, layers={config.gpu_layers}")
        
        # Get estimated memory
        estimated_gb = self._simple_memory_estimate(config)
        print(f"📊 Estimated: {estimated_gb:.2f}GB")
        
        # Check if likely to exceed 48GB
        exceeds_threshold = estimated_gb > 45.0  # Conservative threshold
        if exceeds_threshold:
            print(f"⚠️  Estimated memory {estimated_gb:.2f}GB likely exceeds available VRAM")
        
        # Get baseline memory
        baseline = self._get_baseline_memory()
        print(f"📊 Baseline: GPU0={baseline[0]:.0f}MB, GPU1={baseline[1]:.0f}MB, GPU2={baseline[2]:.0f}MB")
        
        # Build production-like command with all flags as specified
        command = [
            "/llama.cpp/build/bin/llama-server",
            "--host", "127.0.0.1",
            "--port", "8001",
            "--threads", str(config.threads),
            "--ctx-size", str(config.context_size),
            "--batch-size", str(config.batch_size),
            "--ubatch-size", str(config.ubatch_size),
            "--cache-type-k", config.cache_type_k,
            "--cache-type-v", config.cache_type_v,
            "--numa", config.numa,
            "--n-cpu-moe", str(config.n_cpu_moe),
            "--gpu-layers", str(config.gpu_layers),
            "--split-mode", config.split_mode,
            "--tensor-split", config.tensor_split,
            "--main-gpu", str(config.main_gpu),
            "--model", config.gguf_path
        ]
        
        # Add boolean flags
        if config.no_kv_offload:
            command.append("--no-kv-offload")
        if config.no_warmup:
            command.append("--no-warmup")
        if config.cont_batching:
            command.append("--cont-batching")
        if config.no_webui:
            command.append("--no-webui")
        if config.metrics:
            command.append("--metrics")
        if config.jinja:
            command.append("--jinja")
            
        # Add mmproj if present
        if config.mmproj_path:
            command.extend(["--mmproj", config.mmproj_path])
            
        print("🚀 Starting llama-server with full production configuration...")
        
        success = False
        actual_memory = (0.0, 0.0, 0.0)
        error_msg = ""
        oom_detected = False
        
        try:
            # Start process
            process = subprocess.Popen([
                "kubectl", "exec", "-i", "-n", self.namespace, self.pod_name, "--"
            ] + command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True)
            
            print("⏳ Waiting 35 seconds for large model to load...")
            time.sleep(35)
            
            # Check if process is running
            if process.poll() is None:
                actual_memory = self._get_actual_memory()
                print(f"📊 Actual: GPU0={actual_memory[0]:.0f}MB, GPU1={actual_memory[1]:.0f}MB, GPU2={actual_memory[2]:.0f}MB")
                success = True
            else:
                # Process exited early, check for OOM
                stdout, stderr = process.communicate()
                error_output = stderr.strip() if stderr else stdout.strip()
                
                # Check for OOM patterns
                oom_patterns = [
                    "out of memory", "cuda error", "memory allocation", 
                    "cuda out of memory", "insufficient memory", "not enough memory",
                    "failed to allocate", "memory error", "cuda_malloc failed"
                ]
                
                if any(pattern in error_output.lower() for pattern in oom_patterns):
                    oom_detected = True
                    error_msg = f"OOM detected: {error_output[:200]}"
                    print(f"💥 OOM detected: Model requires more than available VRAM")
                else:
                    error_msg = f"Process exited early: {error_output[:200]}"
                    print(f"❌ Process failed: {error_msg}")
                
                actual_memory = self._get_actual_memory()
            
            # Terminate process
            print("🛑 Terminating llama-server...")
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                    print("✅ Server terminated gracefully")
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    print("🔥 Server force-killed")
            
        except Exception as e:
            error_msg = f"Test execution error: {str(e)}"
            print(f"❌ Test failed: {error_msg}")
            actual_memory = self._get_actual_memory()
            
        finally:
            self._cleanup_processes()
            cleanup_memory = self._get_actual_memory()
            print(f"🔍 Post-cleanup memory: GPU0={cleanup_memory[0]:.0f}MB, GPU1={cleanup_memory[1]:.0f}MB, GPU2={cleanup_memory[2]:.0f}MB")
            
        # Calculate results
        actual_memory_mb = [actual_memory[i] - baseline[i] for i in range(3)]
        total_actual_gb = sum(actual_memory_mb) / 1024
        
        accuracy_ratio = total_actual_gb / estimated_gb if estimated_gb > 0 else 0
        exceeds_48gb = total_actual_gb > 48.0 or oom_detected
        
        result = ExactMemoryResult(
            config=config,
            estimated_gb=estimated_gb,
            actual_gpu0_mb=actual_memory_mb[0],
            actual_gpu1_mb=actual_memory_mb[1],
            actual_gpu2_mb=actual_memory_mb[2],
            total_actual_gb=total_actual_gb,
            accuracy_ratio=accuracy_ratio,
            success=success,
            error_msg=error_msg,
            oom_detected=oom_detected,
            exceeds_48gb_vram=exceeds_48gb
        )
        
        if success:
            print(f"✅ Success: {total_actual_gb:.2f}GB actual vs {estimated_gb:.2f}GB estimated ({accuracy_ratio:.2f}x)")
            if exceeds_48gb:
                print("⚠️  Model requires more than 48GB VRAM")
        elif oom_detected:
            print(f"💥 OOM: Model requires more than available VRAM ({estimated_gb:.2f}GB estimated)")
        else:
            print(f"❌ Failed: {error_msg}")
            
        return result
        
    def collect_exact_samples(self):
        """Collect memory samples from the exact requested models"""
        
        print("🎯 Starting Collection for Exact Requested Models")
        print("=" * 80)
        
        # Define tests for EXACT models requested by user
        test_configs = [
            # qwen3-vl-32b-thinking tests
            ExactTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=40960,  # 40K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="32B VL thinking baseline 40K"
            ),
            ExactTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="32B VL thinking 128K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=262144,  # 256K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="32B VL thinking 256K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=524288,  # 512K
                batch_size=1024,
                ubatch_size=1024,
                gpu_layers=-1,
                notes="32B VL thinking 512K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=1000000,  # 1M
                batch_size=512,
                ubatch_size=512,
                gpu_layers=-1,
                notes="32B VL thinking 1M context"
            ),
            
            # qwen3-30b-a3b tests
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=40960,  # 40K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B MoE baseline 40K"
            ),
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B MoE 128K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=262144,  # 256K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="30B MoE 256K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=524288,  # 512K
                batch_size=1024,
                ubatch_size=1024,
                gpu_layers=-1,
                notes="30B MoE 512K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=1000000,  # 1M
                batch_size=512,
                ubatch_size=512,
                gpu_layers=-1,
                notes="30B MoE 1M context"
            ),
            
            # qwen3-coder-30b-a3b tests
            ExactTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B Coder 128K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=262144,  # 256K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B Coder 256K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=524288,  # 512K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="30B Coder 512K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=1000000,  # 1M
                batch_size=512,
                ubatch_size=512,
                gpu_layers=-1,
                notes="30B Coder 1M context"
            ),
            
            # qwen3-vl-30b-a3b-thinking tests
            ExactTestConfiguration(
                model_id="qwen3-vl-30b-a3b-thinking",
                model_name="Qwen3-VL-30B-A3B-Thinking",
                param_size="30B",
                gguf_path="/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf",
                mmproj_path="/models/qwen3-vl-30b-a3b/mmproj.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B VL thinking 128K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-vl-30b-a3b-thinking",
                model_name="Qwen3-VL-30B-A3B-Thinking",
                param_size="30B",
                gguf_path="/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf",
                mmproj_path="/models/qwen3-vl-30b-a3b/mmproj.gguf",
                context_size=262144,  # 256K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B VL thinking 256K context"
            ),
            ExactTestConfiguration(
                model_id="qwen3-vl-30b-a3b-thinking",
                model_name="Qwen3-VL-30B-A3B-Thinking",
                param_size="30B",
                gguf_path="/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf",
                mmproj_path="/models/qwen3-vl-30b-a3b/mmproj.gguf",
                context_size=524288,  # 512K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="30B VL thinking 512K context"
            ),
            
            # GPU layer variations for dynamic handling tests
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=40,  # Partial GPU
                notes="30B MoE partial GPU 40 layers"
            ),
            ExactTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=20,  # Lower GPU
                notes="30B MoE low GPU 20 layers"
            ),
            
            # Batch size variations for production scenarios
            ExactTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=131072,  # 128K
                batch_size=8192,  # High batch
                ubatch_size=8192,
                gpu_layers=-1,
                notes="32B VL thinking high batch 8192"
            ),
            ExactTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=131072,  # 128K
                batch_size=8192,  # High batch
                ubatch_size=8192,
                gpu_layers=-1,
                notes="30B Coder high batch 8192"
            ),
        ]
        
        total_tests = len(test_configs)
        successful_tests = []
        failed_tests = []
        oom_tests = []
        
        print(f"📋 Running {total_tests} tests on exact requested models")
        print("Models: qwen3-vl-32b-thinking, qwen3-30b-a3b, qwen3-coder-30b-a3b, qwen3-vl-30b-a3b-thinking")
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n{'='*70}")
            print(f"🔍 Test {i}/{total_tests} - {config.notes}")
            
            result = self._run_exact_test(config)
            self.results.append(result)
            
            if result.success:
                successful_tests.append(result)
            elif result.oom_detected:
                oom_tests.append(result)
            else:
                failed_tests.append(result)
                
            # Wait between tests
            if i < total_tests:
                print("⏸️  Waiting 10 seconds between tests...")
                time.sleep(10)
                
        # Save and summarize
        self._save_exact_samples()
        self._generate_exact_summary()
        
        return len(successful_tests)
        
    def _save_exact_samples(self):
        """Save exact model samples"""
        
        successful_samples = []
        for result in self.results:
            if result.success:
                sample = {
                    "model_id": result.config.model_id,
                    "model_name": result.config.model_name,
                    "param_size": result.config.param_size,
                    "gguf_path": result.config.gguf_path,
                    "context_size": result.config.context_size,
                    "batch_size": result.config.batch_size,
                    "ubatch_size": result.config.ubatch_size,
                    "gpu_layers": result.config.gpu_layers,
                    "mmproj_path": result.config.mmproj_path,
                    "estimated_gb": result.estimated_gb,
                    "actual_gpu0_mb": result.actual_gpu0_mb,
                    "actual_gpu1_mb": result.actual_gpu1_mb,
                    "actual_gpu2_mb": result.actual_gpu2_mb,
                    "total_actual_gb": result.total_actual_gb,
                    "accuracy_ratio": result.accuracy_ratio,
                    "notes": result.config.notes,
                    "exact_large_model_test": True,  # Flag for exact requested models
                    "production_config": {
                        "threads": result.config.threads,
                        "no_kv_offload": result.config.no_kv_offload,
                        "cache_type_k": result.config.cache_type_k,
                        "cache_type_v": result.config.cache_type_v,
                        "numa": result.config.numa,
                        "n_cpu_moe": result.config.n_cpu_moe,
                        "split_mode": result.config.split_mode,
                        "tensor_split": result.config.tensor_split,
                        "main_gpu": result.config.main_gpu,
                        "cont_batching": result.config.cont_batching
                    }
                }
                successful_samples.append(sample)
                
        # Merge with existing samples
        existing_samples_path = "/Users/lons7862/workspace/llmmllab/inference/test/unit/real_memory_samples.json"
        
        if os.path.exists(existing_samples_path):
            with open(existing_samples_path, 'r', encoding='utf-8') as f:
                existing_samples = json.load(f)
        else:
            existing_samples = []
            
        # Combine samples
        all_samples = existing_samples + successful_samples
        
        # Save combined samples
        with open(existing_samples_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2)
            
        print(f"✅ Added {len(successful_samples)} exact model samples to existing {len(existing_samples)} samples")
        print(f"   Total samples: {len(all_samples)}")
        
        # Save detailed results separately
        exact_model_results_path = "/Users/lons7862/workspace/llmmllab/inference/debug/exact_large_model_results.json"
        with open(exact_model_results_path, 'w', encoding='utf-8') as f:
            results_data = []
            for result in self.results:
                result_dict = {
                    "config": {
                        "model_id": result.config.model_id,
                        "model_name": result.config.model_name,
                        "param_size": result.config.param_size,
                        "gguf_path": result.config.gguf_path,
                        "context_size": result.config.context_size,
                        "batch_size": result.config.batch_size,
                        "ubatch_size": result.config.ubatch_size,
                        "gpu_layers": result.config.gpu_layers,
                        "mmproj_path": result.config.mmproj_path,
                        "notes": result.config.notes
                    },
                    "estimated_gb": result.estimated_gb,
                    "actual_gpu0_mb": result.actual_gpu0_mb,
                    "actual_gpu1_mb": result.actual_gpu1_mb,
                    "actual_gpu2_mb": result.actual_gpu2_mb,
                    "total_actual_gb": result.total_actual_gb,
                    "accuracy_ratio": result.accuracy_ratio,
                    "success": result.success,
                    "error_msg": result.error_msg,
                    "oom_detected": result.oom_detected,
                    "exceeds_48gb_vram": result.exceeds_48gb_vram
                }
                results_data.append(result_dict)
            json.dump(results_data, f, indent=2)
            
        print(f"✅ Saved detailed exact model results to {exact_model_results_path}")
        
    def _generate_exact_summary(self):
        """Generate summary for exact model results"""
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success and not r.oom_detected]
        oom_results = [r for r in self.results if r.oom_detected]
        
        print("\n" + "="*80)
        print("📊 EXACT LARGE MODEL MEMORY COLLECTION RESULTS")
        print("="*80)
        
        print(f"\n📈 Summary: {len(successful_results)} successful, {len(failed_results)} failed, {len(oom_results)} OOM")
        print("Models tested: qwen3-vl-32b-thinking, qwen3-30b-a3b, qwen3-coder-30b-a3b, qwen3-vl-30b-a3b-thinking")
        
        if successful_results:
            print("\n✅ Successfully Collected Exact Model Samples:")
            print("Model                          Context    Batch    Est GB   Act GB   Accuracy")
            print("-" * 85)
            for result in successful_results:
                ctx_str = f"{result.config.context_size//1024}K" if result.config.context_size >= 1024 else str(result.config.context_size)
                mmproj_indicator = "+VL" if result.config.mmproj_path else ""
                model_display = f"{result.config.model_name[:22]}{mmproj_indicator}"
                print(f"{model_display:<30} {ctx_str:>8} {result.config.batch_size:>8} {result.estimated_gb:>8.1f} {result.total_actual_gb:>8.1f} {result.accuracy_ratio:>8.2f}")
                
        if oom_results:
            print("\n💥 OOM Tests (Require >48GB VRAM):")
            for result in oom_results:
                ctx_str = f"{result.config.context_size//1024}K" if result.config.context_size >= 1024 else str(result.config.context_size)
                print(f"- {result.config.model_name} @ {ctx_str}: {result.estimated_gb:.1f}GB estimated")
                
        if failed_results:
            print("\n❌ Failed Tests:")
            for result in failed_results:
                ctx_str = f"{result.config.context_size//1024}K" if result.config.context_size >= 1024 else str(result.config.context_size)
                print(f"- {result.config.model_name} @ {ctx_str}: {result.error_msg[:80]}")
                
        if successful_results:
            accuracy_ratios = [r.accuracy_ratio for r in successful_results]
            memory_usages = [r.total_actual_gb for r in successful_results]
            
            print("\nExact Model Accuracy Statistics:")
            print(f"Average: {sum(accuracy_ratios)/len(accuracy_ratios):.2f}x")
            print(f"Range: {min(accuracy_ratios):.2f}x - {max(accuracy_ratios):.2f}x")
            
            print(f"\nActual Memory Usage:")
            print(f"Min: {min(memory_usages):.1f}GB")
            print(f"Max: {max(memory_usages):.1f}GB")
            print(f"Average: {sum(memory_usages)/len(memory_usages):.1f}GB")


if __name__ == "__main__":
    # Get pod name
    pod_result = subprocess.run([
        "kubectl", "get", "pods", "-n", "ollama", 
        "-o", "jsonpath={.items[0].metadata.name}"
    ], capture_output=True, text=True, check=False)
    
    if pod_result.returncode != 0:
        print("❌ Failed to get pod name")
        exit(1)
        
    pod_name = pod_result.stdout.strip()
    print(f"🎯 Using pod: {pod_name}")
    
    collector = ExactModelSampleCollector(k8s_pod_name=pod_name)
    collector.collect_exact_samples()