#!/usr/bin/env python3
"""
Focused Large Model Memory Collection

Collects memory samples from available large models with realistic configurations
that will work within current VRAM constraints.
"""

import subprocess
import json
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class FocusedTestConfiguration:
    """Focused test configuration for memory measurement"""
    
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


@dataclass 
class FocusedMemoryResult:
    """Memory measurement result"""
    
    config: FocusedTestConfiguration
    estimated_gb: float
    actual_gpu0_mb: float
    actual_gpu1_mb: float
    actual_gpu2_mb: float
    total_actual_gb: float
    accuracy_ratio: float
    success: bool
    error_msg: str = ""


class FocusedModelSampleCollector:
    """Collects memory samples with focused, realistic configurations"""
    
    def __init__(self, k8s_pod_name: str):
        self.pod_name = k8s_pod_name
        self.namespace = "ollama"
        self.results: List[FocusedMemoryResult] = []
        
    def _get_memory_estimate_from_container(self, config: FocusedTestConfiguration) -> float:
        """Get memory estimate using container script"""
        try:
            cmd = [
                "kubectl", "exec", "-n", self.namespace, self.pod_name, "--",
                "python3", "/app/estimate_memory.py",
                "--model", config.gguf_path,
                "--ctx-size", str(config.context_size),
                "--batch-size", str(config.batch_size),
                "--gpu-layers", str(config.gpu_layers)
            ]
            
            if config.mmproj_path:
                cmd.extend(["--mmproj", config.mmproj_path])
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
            
            if result.returncode != 0:
                print(f"❌ Estimation failed: {result.stderr.strip()}")
                # Use fallback based on model name
                if "30b" in config.model_name.lower() or "32b" in config.model_name.lower():
                    return 25.0  # Large model fallback
                elif "4b" in config.model_name.lower():
                    return 12.0  # Medium model fallback 
                else:
                    return 8.0   # Small model fallback
                    
            # Parse output for memory estimate
            for line in result.stdout.strip().split('\n'):
                if "Total GPU Memory:" in line:
                    gb_str = line.split(":")[-1].strip().replace("GB", "").strip()
                    return float(gb_str)
                    
            return 20.0  # Parse fallback
            
        except Exception as e:
            print(f"❌ Memory estimation error: {e}")
            return 20.0
            
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
                
        except Exception as e:
            print(f"🧹 Cleanup warning: {e}")
            
    def _run_focused_test(self, config: FocusedTestConfiguration) -> FocusedMemoryResult:
        """Run focused memory test"""
        
        # Format context size
        if config.context_size >= 1048576:
            ctx_display = f"{config.context_size//1048576}M"
        elif config.context_size >= 1024:
            ctx_display = f"{config.context_size//1024}K"
        else:
            ctx_display = f"{config.context_size}"
            
        print(f"\n🧪 Testing {config.model_name} @ {ctx_display} ctx, batch={config.batch_size}, layers={config.gpu_layers}")
        
        # Get estimated memory
        estimated_gb = self._get_memory_estimate_from_container(config)
        print(f"📊 Estimated: {estimated_gb:.2f}GB")
        
        # Get baseline memory
        baseline = self._get_baseline_memory()
        print(f"📊 Baseline: GPU0={baseline[0]:.0f}MB, GPU1={baseline[1]:.0f}MB, GPU2={baseline[2]:.0f}MB")
        
        # Build simpler command for testing
        command = [
            "/llama.cpp/build/bin/llama-server",
            "--host", "0.0.0.0",
            "--port", "8080",
            "--ctx-size", str(config.context_size),
            "--batch-size", str(config.batch_size), 
            "--n-gpu-layers", str(config.gpu_layers),
            "--model", config.gguf_path
        ]
        
        if config.mmproj_path:
            command.extend(["--mmproj", config.mmproj_path])
            
        print("🚀 Starting llama-server...")
        
        success = False
        actual_memory = (0.0, 0.0, 0.0)
        error_msg = ""
        
        try:
            # Start process
            process = subprocess.Popen([
                "kubectl", "exec", "-i", "-n", self.namespace, self.pod_name, "--"
            ] + command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True)
            
            print("⏳ Waiting 25 seconds for model to load...")
            time.sleep(25)
            
            # Check if process is running
            if process.poll() is None:
                actual_memory = self._get_actual_memory()
                print(f"📊 Actual: GPU0={actual_memory[0]:.0f}MB, GPU1={actual_memory[1]:.0f}MB, GPU2={actual_memory[2]:.0f}MB")
                success = True
            else:
                stdout, stderr = process.communicate()
                error_output = stderr.strip() if stderr else stdout.strip()
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
        
        result = FocusedMemoryResult(
            config=config,
            estimated_gb=estimated_gb,
            actual_gpu0_mb=actual_memory_mb[0],
            actual_gpu1_mb=actual_memory_mb[1],
            actual_gpu2_mb=actual_memory_mb[2],
            total_actual_gb=total_actual_gb,
            accuracy_ratio=accuracy_ratio,
            success=success,
            error_msg=error_msg
        )
        
        if success:
            print(f"✅ Success: {total_actual_gb:.2f}GB actual vs {estimated_gb:.2f}GB estimated ({accuracy_ratio:.2f}x)")
        else:
            print(f"❌ Failed: {error_msg}")
            
        return result
        
    def collect_focused_samples(self):
        """Collect focused memory samples from available large models"""
        
        print("🎯 Starting Focused Large Model Memory Collection")
        print("=" * 80)
        
        # Focused test configurations - realistic for current hardware
        test_configs = [
            # Qwen3-VL-2B tests (should work, baseline for VL models)
            FocusedTestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B-Thinking",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                context_size=40960,  # 40K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="2B VL baseline with mmproj"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B-Thinking",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="2B VL high context 128K"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B-Thinking",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                context_size=262144,  # 256K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="2B VL very high context 256K"
            ),
            
            # Qwen3-4B tests (larger context, production batch sizes)
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=40960,  # 40K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="4B text model high context 40K, production batch"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="4B text model very high context 128K"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=262144,  # 256K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="4B text model extreme context 256K"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=524288,  # 512K
                batch_size=1024,
                ubatch_size=1024,
                gpu_layers=-1,
                notes="4B text model massive context 512K"
            ),
            
            # Production batch size variations
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=131072,  # 128K
                batch_size=8192,
                ubatch_size=8192,
                gpu_layers=-1,
                notes="4B text model high batch 8192"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=131072,  # 128K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="4B text model medium batch 2048"
            ),
            
            # GPU layer variations (partial offloading scenarios)
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=30,  # Partial GPU
                notes="4B partial GPU offload 30 layers"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=20,  # Lower GPU
                notes="4B low GPU offload 20 layers"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=10,  # Minimal GPU
                notes="4B minimal GPU offload 10 layers"
            ),
            
            # Try one larger model if available (may fail, that's OK)
            FocusedTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=40960,  # 40K (conservative)
                batch_size=2048,  # Conservative batch
                ubatch_size=2048,
                gpu_layers=20,  # Conservative GPU layers
                notes="30B MoE conservative test (may fail)"
            ),
            FocusedTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=40960,  # 40K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=10,  # Very conservative
                notes="30B MoE very conservative GPU (may fail)"
            ),
        ]
        
        total_tests = len(test_configs)
        successful_tests = []
        failed_tests = []
        
        print(f"📋 Running {total_tests} focused large model tests")
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n{'='*60}")
            print(f"🔍 Test {i}/{total_tests} - {config.notes}")
            
            result = self._run_focused_test(config)
            self.results.append(result)
            
            if result.success:
                successful_tests.append(result)
            else:
                failed_tests.append(result)
                
            # Wait between tests
            if i < total_tests:
                print("⏸️  Waiting 8 seconds between tests...")
                time.sleep(8)
                
        # Save and summarize
        self._save_focused_samples()
        self._generate_focused_summary()
        
        return len(successful_tests)
        
    def _save_focused_samples(self):
        """Save focused samples"""
        
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
                    "large_model_test": True,  # Flag to identify these as large model tests
                    "production_focused": True
                }
                successful_samples.append(sample)
                
        # Merge with existing samples
        import os
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
            
        print(f"✅ Added {len(successful_samples)} large model samples to existing {len(existing_samples)} samples")
        print(f"   Total samples: {len(all_samples)}")
        
    def _generate_focused_summary(self):
        """Generate summary"""
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print("\n" + "="*80)
        print("📊 FOCUSED LARGE MODEL MEMORY COLLECTION RESULTS")
        print("="*80)
        
        print(f"\n📈 Summary: {len(successful_results)} successful, {len(failed_results)} failed")
        
        if successful_results:
            print("\n✅ Successfully Collected Samples:")
            print("Model                     Context    Batch    Est GB   Act GB   Accuracy")
            print("-" * 75)
            for result in successful_results:
                ctx_str = f"{result.config.context_size//1024}K"
                print(f"{result.config.model_name[:20]:<20} {ctx_str:>8} {result.config.batch_size:>8} {result.estimated_gb:>8.1f} {result.total_actual_gb:>8.1f} {result.accuracy_ratio:>8.2f}")
                
        if failed_results:
            print("\n❌ Failed Tests:")
            for result in failed_results:
                print(f"- {result.config.model_name} @ {result.config.context_size//1024}K: {result.error_msg[:60]}")
                
        if successful_results:
            accuracy_ratios = [r.accuracy_ratio for r in successful_results]
            print("\nAccuracy Statistics:")
            print(f"Average: {sum(accuracy_ratios)/len(accuracy_ratios):.2f}x")
            print(f"Range: {min(accuracy_ratios):.2f}x - {max(accuracy_ratios):.2f}x")


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
    
    collector = FocusedModelSampleCollector(k8s_pod_name=pod_name)
    collector.collect_focused_samples()