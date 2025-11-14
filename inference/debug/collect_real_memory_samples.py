#!/usr/bin/env python3
"""
Real Memory Sample Collector

This script runs a focused set of real memory tests to gather actual data
for comprehensive unit test validation.
"""

import subprocess
import time
import json
from typing import Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestConfiguration:
    """Test configuration for memory measurement"""

    model_id: str
    model_name: str
    param_size: str
    gguf_path: str
    context_size: int
    batch_size: int
    gpu_layers: int
    mmproj_path: Optional[str] = None
    notes: str = ""


@dataclass
class RealMemoryResult:
    """Real memory measurement result"""

    config: TestConfiguration
    estimated_gb: float
    actual_gpu0_mb: float
    actual_gpu1_mb: float
    actual_gpu2_mb: float
    total_actual_gb: float
    accuracy_ratio: float
    success: bool
    error_msg: str = ""


class RealMemorySampleCollector:

    def __init__(self, k8s_pod_name: str = "ollama-5567bf7859-rwj6c"):
        self.pod_name = k8s_pod_name
        self.namespace = "ollama"
        self.results: list[RealMemoryResult] = []

    def _get_memory_estimate_from_container(self, config: TestConfiguration) -> float:
        """Get memory estimate by running estimation script inside container"""
        try:
            # Create temporary estimation script
            estimate_script = f"""
import sys
sys.path.insert(0, "/app")

from models import Model, ModelProfile, ModelParameters, ModelDetails, ModelProvider, ModelTask
from runner.pipeline_cache import LocalPipelineCacheManager

model = Model(
    id="{config.model_id}",
    name="{config.model_name}",
    model="{config.gguf_path}",
    provider=ModelProvider.LLAMA_CPP,
    task=ModelTask.TEXTTOTEXT,
    modified_at="2025-11-13",
    digest="test-digest",
    size=1000000000,
    details=ModelDetails(
        format="gguf",
        family="test",
        families=["test"],
        parameter_size="{config.param_size}",
        quantization_level="q4_k_m",
        size=1000000000,
        original_ctx=4096,
        gguf_file="{config.gguf_path}",
        clip_model_path={'"' + config.mmproj_path + '"' if config.mmproj_path else 'None'},
    ),
)

profile = ModelProfile(
    user_id="test_user",
    name="Test {config.context_size//1024}K",
    model_name="{config.model_id}",
    system_prompt="Test",
    type=1,
    parameters=ModelParameters(
        num_ctx={config.context_size},
        batch_size={config.batch_size},
        n_gpu_layers={config.gpu_layers},
    )
)

cache_manager = LocalPipelineCacheManager()
estimated_bytes = cache_manager.estimate_memory(model, profile)
print(f"ESTIMATE_GB:{{estimated_bytes / (1024**3):.3f}}")
"""

            # Write script to container
            write_process = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    "-i",
                    "-n",
                    self.namespace,
                    self.pod_name,
                    "--",
                    "tee",
                    "/tmp/sample_estimate.py",
                ],
                input=estimate_script,
                text=True,
                capture_output=True,
                check=False,
            )

            if write_process.returncode != 0:
                print(f"❌ Failed to write estimation script: {write_process.stderr}")
                return 0.0

            # Run estimation script
            result = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    self.pod_name,
                    "--",
                    "/app/v.sh",
                    "python",
                    "/tmp/sample_estimate.py",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if result.returncode != 0:
                print(f"❌ Estimation script failed: {result.stderr}")
                return 0.0

            # Extract the estimate marked with ESTIMATE_GB:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("ESTIMATE_GB:"):
                    estimate_str = line.replace("ESTIMATE_GB:", "")
                    try:
                        return float(estimate_str)
                    except ValueError:
                        print(f"❌ Could not parse estimate: '{estimate_str}'")

            print("❌ No valid estimate found in output")
            return 0.0

        except Exception as e:
            print(f"❌ Failed to get memory estimate: {e}")
            return 0.0

    def _get_baseline_memory(self) -> Tuple[float, float, float]:
        """Get baseline GPU memory usage"""
        try:
            result = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    self.pod_name,
                    "--",
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )

            # Parse memory values from output
            lines = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
            memory_values = []

            for line in lines:
                try:
                    memory_values.append(float(line))
                except ValueError:
                    print(f"⚠️  Could not parse memory value: '{line}'")

            # Ensure we have at least 3 values, pad with 0 if needed
            while len(memory_values) < 3:
                memory_values.append(0.0)

            return tuple(memory_values[:3])  # Return first 3 GPU values

        except Exception as e:
            print(f"❌ Failed to get baseline memory: {e}")
            return (0.0, 0.0, 0.0)

    def _run_real_memory_test(self, config: TestConfiguration) -> RealMemoryResult:
        """Run real memory test for a single configuration"""

        # Format context size nicely for display
        if config.context_size >= 1048576:
            ctx_display = f"{config.context_size//1048576}M"
        elif config.context_size >= 1024:
            ctx_display = f"{config.context_size//1024}K"
        else:
            ctx_display = str(config.context_size)

        print(
            f"\n🧪 Testing {config.model_name} @ {ctx_display} ctx, batch={config.batch_size}, layers={config.gpu_layers}"
        )

        # Get estimated memory
        estimated_gb = self._get_memory_estimate_from_container(config)
        print(f"📊 Estimated: {estimated_gb:.2f}GB")

        # Get baseline memory
        baseline = self._get_baseline_memory()
        print(
            f"📊 Baseline: GPU0={baseline[0]:.0f}MB, GPU1={baseline[1]:.0f}MB, GPU2={baseline[2]:.0f}MB"
        )

        # Build command
        command = [
            "/llama.cpp/build/bin/llama-server",
            "--model",
            config.gguf_path,
            "--ctx-size",
            str(config.context_size),
            "--batch-size",
            str(config.batch_size),
            "--n-gpu-layers",
            str(config.gpu_layers),
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]

        if config.mmproj_path:
            command.extend(["--mmproj", config.mmproj_path])

        print("🚀 Starting llama-server...")

        success = False
        actual_memory = (0.0, 0.0, 0.0)
        error_msg = ""

        try:
            # Start llama-server in background
            process = subprocess.Popen(
                ["kubectl", "exec", "-n", self.namespace, self.pod_name, "--"]
                + command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Wait for model to load
            print("⏳ Waiting 25 seconds for model to load...")
            time.sleep(25)

            # Check if process is still running
            if process.poll() is not None:
                _, stderr = process.communicate()
                error_msg = (
                    f"Process exited early: {stderr[-500:]}"  # Last 500 chars of stderr
                )
                print(f"❌ Process failed: {error_msg}")
            else:
                # Get memory usage
                actual_memory = self._get_baseline_memory()
                success = True
                print(
                    f"📊 Actual: GPU0={actual_memory[0]:.0f}MB, GPU1={actual_memory[1]:.0f}MB, GPU2={actual_memory[2]:.0f}MB"
                )

            # Kill the server with comprehensive cleanup
            try:
                print("🛑 Terminating llama-server...")
                process.terminate()
                try:
                    process.wait(timeout=8)
                    print("✅ Server terminated gracefully")
                except subprocess.TimeoutExpired:
                    print("🔪 Force killing llama-server...")
                    process.kill()
                    process.wait(timeout=3)
                    print("✅ Server force-killed")
            except Exception as shutdown_error:
                print(f"⚠️  Error during server shutdown: {shutdown_error}")
            
            # Additional cleanup - kill any remaining llama-server processes
            try:
                result = subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        self.namespace,
                        self.pod_name,
                        "--",
                        "pkill",
                        "-f",
                        "llama-server",
                    ],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode == 0:
                    print("🧹 Cleaned up any remaining llama-server processes")
                else:
                    print("🧹 No additional llama-server processes found")
            except subprocess.TimeoutExpired:
                print("⚠️  Cleanup command timed out")

            # Wait for complete cleanup and verify
            time.sleep(4)
            
            # Verify cleanup with memory check
            try:
                final_memory = self._get_baseline_memory()
                print(f"🔍 Post-cleanup memory: GPU0={final_memory[0]:.0f}MB, GPU1={final_memory[1]:.0f}MB, GPU2={final_memory[2]:.0f}MB")
            except:
                print("⚠️  Could not verify final memory state")

        except Exception as e:
            error_msg = str(e)
            print(f"❌ Test failed: {e}")
            # Ensure cleanup even if test failed
            try:
                subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        self.namespace,
                        self.pod_name,
                        "--",
                        "pkill",
                        "-f",
                        "llama-server",
                    ],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                pass

        # Calculate results
        actual_memory_mb = [actual_memory[i] - baseline[i] for i in range(3)]
        total_actual_gb = sum(actual_memory_mb) / 1024

        accuracy_ratio = total_actual_gb / estimated_gb if estimated_gb > 0 else 0

        result = RealMemoryResult(
            config=config,
            estimated_gb=estimated_gb,
            actual_gpu0_mb=actual_memory_mb[0],
            actual_gpu1_mb=actual_memory_mb[1],
            actual_gpu2_mb=actual_memory_mb[2],
            total_actual_gb=total_actual_gb,
            accuracy_ratio=accuracy_ratio,
            success=success,
            error_msg=error_msg,
        )

        if success:
            print(
                f"✅ Success: {total_actual_gb:.2f}GB actual vs {estimated_gb:.2f}GB estimated ({accuracy_ratio:.2f}x)"
            )
        else:
            print(f"❌ Failed: {error_msg[:100]}")

        return result

    def collect_real_samples(self):
        """Collect real memory samples from diverse configurations"""

        print("🎯 Starting Real Memory Sample Collection")
        print("=" * 80)

        # Define diverse test configurations - comprehensive coverage (25+ tests)
        test_configs = [
            # === SMALL MODEL TESTS (3.2B Llama) ===
            # Baseline small contexts
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=4096,   # 4K
                batch_size=512,
                gpu_layers=35,
                notes="3.2B baseline 4K context",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=8192,   # 8K
                batch_size=512,
                gpu_layers=35,
                notes="3.2B baseline 8K context",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=16384,  # 16K
                batch_size=512,
                gpu_layers=35,
                notes="3.2B medium 16K context",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=32768,  # 32K
                batch_size=256,
                gpu_layers=35,
                notes="3.2B large 32K context",
            ),
            
            # Large contexts for 3.2B
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=65536,  # 64K
                batch_size=256,
                gpu_layers=25,
                notes="3.2B very large 64K context",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=131072, # 128K
                batch_size=256,
                gpu_layers=20,
                notes="3.2B extreme 128K context",
            ),
            
            # Batch size variations for 3.2B
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=8192,
                batch_size=128,      # Very small batch
                gpu_layers=35,
                notes="3.2B tiny batch 128",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=8192,
                batch_size=1024,     # Large batch
                gpu_layers=35,
                notes="3.2B large batch 1024",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=8192,
                batch_size=2048,     # Very large batch
                gpu_layers=25,
                notes="3.2B huge batch 2048",
            ),
            
            # GPU layer variations for 3.2B
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=8192,
                batch_size=512,
                gpu_layers=10,       # Few GPU layers
                notes="3.2B few GPU layers 10",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=8192,
                batch_size=512,
                gpu_layers=50,       # Many GPU layers
                notes="3.2B many GPU layers 50",
            ),
            
            # === MEDIUM MODEL TESTS (4B Qwen3) ===
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=4096,   # 4K
                batch_size=512,
                gpu_layers=25,
                notes="4B baseline 4K context",
            ),
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=8192,   # 8K
                batch_size=512,
                gpu_layers=25,
                notes="4B baseline 8K context",
            ),
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=16384,  # 16K
                batch_size=512,
                gpu_layers=35,
                notes="4B medium 16K context",
            ),
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=32768,  # 32K
                batch_size=256,
                gpu_layers=25,
                notes="4B large 32K context",
            ),
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=8192,
                batch_size=1024,     # Large batch
                gpu_layers=35,
                notes="4B large batch 1024",
            ),
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=8192,
                batch_size=256,      # Small batch
                gpu_layers=35,
                notes="4B small batch 256",
            ),
            
            # === VISION MODEL TESTS (2B with mmproj) ===
            TestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                context_size=4096,   # 4K
                batch_size=512,
                gpu_layers=25,
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                notes="VL-2B with mmproj 4K context",
            ),
            TestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                context_size=8192,   # 8K
                batch_size=512,
                gpu_layers=25,
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                notes="VL-2B with mmproj 8K context",
            ),
            TestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                context_size=16384,  # 16K
                batch_size=256,
                gpu_layers=25,
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                notes="VL-2B with mmproj 16K context",
            ),
            TestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                context_size=8192,
                batch_size=1024,     # Large batch
                gpu_layers=35,
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                notes="VL-2B with mmproj large batch",
            ),
            
            # === EXTREME CONTEXT TESTS (estimation validation) ===
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=262144, # 256K
                batch_size=256,
                gpu_layers=20,
                notes="3.2B extreme 256K context",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=524288, # 512K
                batch_size=128,
                gpu_layers=15,
                notes="3.2B massive 512K context",
            ),
            TestConfiguration(
                model_id="llama-chat-summary-3_2-3b",
                model_name="Llama 3.2 3B",
                param_size="3.2B",
                gguf_path="/models/llama-chat-summary/llama-chat-summary-3.2-3b-q5_k_m.gguf",
                context_size=1048576,# 1M
                batch_size=64,
                gpu_layers=10,
                notes="3.2B ultimate 1M context",
            ),
            
            # === MIXED PARAMETER STRESS TESTS ===
            TestConfiguration(
                model_id="qwen3-4b",
                model_name="Qwen3-4B",
                param_size="4B",
                gguf_path="/models/qwen3-4b/qwen3-4b-ud-q6-k-xl.gguf",
                context_size=65536,  # 64K
                batch_size=128,      # Small batch for large context
                gpu_layers=20,
                notes="4B stress test 64K + small batch",
            ),
            TestConfiguration(
                model_id="qwen3-vl-2b-thinking",
                model_name="Qwen3-VL-2B",
                param_size="2B",
                gguf_path="/models/qwen3-vl-2b/qwen3-vl-2b-thinking-abliterated.gguf",
                context_size=32768,  # 32K
                batch_size=256,
                gpu_layers=15,       # Fewer layers
                mmproj_path="/models/qwen3-vl-2b/mmproj.gguf",
                notes="VL-2B stress test 32K + fewer layers",
            ),
        ]

        print(f"📋 Running {len(test_configs)} real memory tests")

        for i, config in enumerate(test_configs):
            print(f"\n{'='*60}")
            print(f"🔍 Test {i+1}/{len(test_configs)} - {config.notes}")

            result = self._run_real_memory_test(config)
            self.results.append(result)

            # Wait between tests for cleanup
            if i < len(test_configs) - 1:
                print("⏸️  Waiting 8 seconds between tests...")
                time.sleep(8)

        self._save_real_samples()
        self._generate_summary()

    def _save_real_samples(self):
        """Save collected real samples for unit test use"""

        # Convert to serializable format
        samples_data = []
        for result in self.results:
            if result.success:
                sample = {
                    "model_id": result.config.model_id,
                    "model_name": result.config.model_name,
                    "param_size": result.config.param_size,
                    "gguf_path": result.config.gguf_path,
                    "context_size": result.config.context_size,
                    "batch_size": result.config.batch_size,
                    "gpu_layers": result.config.gpu_layers,
                    "mmproj_path": result.config.mmproj_path,
                    "estimated_gb": result.estimated_gb,
                    "actual_gpu0_mb": result.actual_gpu0_mb,
                    "actual_gpu1_mb": result.actual_gpu1_mb,
                    "actual_gpu2_mb": result.actual_gpu2_mb,
                    "total_actual_gb": result.total_actual_gb,
                    "accuracy_ratio": result.accuracy_ratio,
                    "notes": result.config.notes,
                }
                samples_data.append(sample)

        # Save to file for unit tests to use
        samples_path = "/Users/lons7862/workspace/llmmllab/inference/test/unit/real_memory_samples.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2)

        print(f"\n✅ Saved {len(samples_data)} real samples to {samples_path}")

        # Also save comprehensive results
        all_results_path = "/Users/lons7862/workspace/llmmllab/inference/debug/real_memory_collection_results.json"
        all_results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert config to dict manually since it's nested
            result_dict["config"] = asdict(result.config)
            all_results_data.append(result_dict)

        with open(all_results_path, "w", encoding="utf-8") as f:
            json.dump(all_results_data, f, indent=2)

        print(f"✅ Saved all {len(all_results_data)} results to {all_results_path}")

    def _generate_summary(self):
        """Generate summary of sample collection"""

        print(f"\n{'='*80}")
        print("📊 REAL MEMORY SAMPLE COLLECTION RESULTS")
        print("=" * 80)

        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]

        print(
            f"\n📈 Summary: {len(successful_results)} successful, {len(failed_results)} failed"
        )

        if successful_results:
            print("\n✅ Successfully Collected Samples:")
            print(
                f"{'Model':<25} {'Context':<10} {'Batch':<8} {'Est GB':<8} {'Act GB':<8} {'Accuracy':<8}"
            )
            print("-" * 85)

            for result in successful_results:
                config = result.config
                # Format context size nicely
                if config.context_size >= 1048576:
                    ctx_display = f"{config.context_size//1048576}M"
                elif config.context_size >= 1024:
                    ctx_display = f"{config.context_size//1024}K"
                else:
                    ctx_display = str(config.context_size)

                print(
                    f"{config.model_name[:24]:<25} {ctx_display:<10} "
                    f"{config.batch_size:<8} {result.estimated_gb:<8.2f} {result.total_actual_gb:<8.2f} {result.accuracy_ratio:<8.2f}"
                )

        if failed_results:
            print("\n❌ Failed Tests:")
            for result in failed_results:
                config = result.config
                ctx_display = (
                    f"{config.context_size//1024}K"
                    if config.context_size >= 1024
                    else str(config.context_size)
                )
                print(f"- {config.model_name} @ {ctx_display}: {result.error_msg[:60]}")

        # Calculate accuracy statistics
        if successful_results:
            accuracies = [r.accuracy_ratio for r in successful_results]
            avg_accuracy = sum(accuracies) / len(accuracies)
            min_accuracy = min(accuracies)
            max_accuracy = max(accuracies)

            print("\n📈 Accuracy Statistics:")
            print(f"Average: {avg_accuracy:.2f}x")
            print(f"Min: {min_accuracy:.2f}x")
            print(f"Max: {max_accuracy:.2f}x")
            print(f"Range: {min_accuracy:.2f}x - {max_accuracy:.2f}x")


if __name__ == "__main__":
    # Get pod name dynamically
    pod_result = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            "ollama",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if pod_result.returncode != 0:
        print("❌ Failed to get pod name")
        exit(1)

    pod_name = pod_result.stdout.strip()
    print(f"🎯 Using pod: {pod_name}")

    collector = RealMemorySampleCollector(k8s_pod_name=pod_name)
    collector.collect_real_samples()
