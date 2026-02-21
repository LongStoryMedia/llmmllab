#!/usr/bin/env python3
"""
Large Model Real Memory Sample Collector

This script collects real memory measurements from large models (30B+, VL models)
for comprehensive unit test validation with production-like configurations.
"""

import subprocess
import json
import time
import signal
from typing import Tuple, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class LargeModelTestConfiguration:
    """Test configuration for large model memory measurement"""

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
    # Production-like parameters
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
class LargeModelMemoryResult:
    """Real memory measurement result for large models"""

    config: LargeModelTestConfiguration
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


class LargeModelSampleCollector:
    """Collects memory samples from large models with production-like configurations"""

    def __init__(self, k8s_pod_name: str = "ollama-5567bf7859-rwj6c"):
        self.pod_name = k8s_pod_name
        self.namespace = "llmmll"
        self.results: List[LargeModelMemoryResult] = []

    def _get_memory_estimate_from_container(
        self, config: LargeModelTestConfiguration
    ) -> float:
        """Get memory estimate by running estimation script inside container"""
        try:
            # Build command to run estimation script
            cmd = [
                "kubectl",
                "exec",
                "-n",
                self.namespace,
                self.pod_name,
                "--",
                "python3",
                "/app/estimate_memory.py",
                "--model",
                config.gguf_path,
                "--ctx-size",
                str(config.context_size),
                "--batch-size",
                str(config.batch_size),
                "--gpu-layers",
                str(config.gpu_layers),
            ]

            if config.mmproj_path:
                cmd.extend(["--mmproj", config.mmproj_path])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            )

            if result.returncode != 0:
                print(f"❌ Estimation failed: {result.stderr}")
                return 8.0  # Fallback estimate

            # Parse output for memory estimate
            lines = result.stdout.strip().split("\n")
            for line in lines:
                if "Total GPU Memory:" in line:
                    # Extract GB value
                    gb_str = line.split(":")[-1].strip().replace("GB", "").strip()
                    return float(gb_str)

            return 8.0  # Fallback if parsing fails

        except Exception as e:
            print(f"❌ Memory estimation error: {e}")
            return 8.0

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
                timeout=10,
                check=True,
            )

            memory_values = [
                float(x.strip()) for x in result.stdout.strip().split("\n")
            ]

            # Ensure we have 3 values (pad with 0 if fewer GPUs)
            while len(memory_values) < 3:
                memory_values.append(0.0)

            return tuple(memory_values[:3])

        except Exception as e:
            print(f"❌ Failed to get baseline memory: {e}")
            return (4.0, 4.0, 4.0)  # Fallback baseline

    def _get_actual_memory(self) -> Tuple[float, float, float]:
        """Get actual GPU memory usage"""
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
                timeout=10,
                check=True,
            )

            memory_values = [
                float(x.strip()) for x in result.stdout.strip().split("\n")
            ]

            # Ensure we have 3 values
            while len(memory_values) < 3:
                memory_values.append(0.0)

            return tuple(memory_values[:3])

        except Exception as e:
            print(f"❌ Failed to get actual memory: {e}")
            return (0.0, 0.0, 0.0)

    def _cleanup_processes(self):
        """Clean up any remaining llama-server processes"""
        try:
            # Kill any existing llama-server processes
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

            # Wait a moment for cleanup
            time.sleep(3)

            # Verify cleanup
            result = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    "-n",
                    self.namespace,
                    self.pod_name,
                    "--",
                    "pgrep",
                    "-f",
                    "llama-server",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                print(f"🧹 Found remaining processes: {result.stdout.strip()}")
                # Force kill with SIGKILL
                subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        self.namespace,
                        self.pod_name,
                        "--",
                        "pkill",
                        "-9",
                        "-f",
                        "llama-server",
                    ],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                time.sleep(2)
            else:
                print("🧹 No additional llama-server processes found")

        except Exception as e:
            print(f"🧹 Cleanup error (non-critical): {e}")

    def _check_memory_threshold(self, memory_gb: float) -> bool:
        """Check if memory usage exceeds 48GB VRAM threshold"""
        return memory_gb > 48.0

    def _run_large_model_test(
        self, config: LargeModelTestConfiguration
    ) -> LargeModelMemoryResult:
        """Run real memory test for a large model configuration"""

        # Format context size nicely for display
        if config.context_size >= 1048576:
            ctx_display = f"{config.context_size//1048576}M"
        elif config.context_size >= 1024:
            ctx_display = f"{config.context_size//1024}K"
        else:
            ctx_display = f"{config.context_size}"

        print(
            f"\n🧪 Testing {config.model_name} @ {ctx_display} ctx, batch={config.batch_size}, ubatch={config.ubatch_size}, layers={config.gpu_layers}"
        )

        # Get estimated memory
        estimated_gb = self._get_memory_estimate_from_container(config)
        print(f"📊 Estimated: {estimated_gb:.2f}GB")

        # Check if estimate exceeds threshold
        exceeds_threshold = self._check_memory_threshold(estimated_gb)
        if exceeds_threshold:
            print(f"⚠️  Estimated memory {estimated_gb:.2f}GB exceeds 48GB VRAM limit")

        # Get baseline memory
        baseline = self._get_baseline_memory()
        print(
            f"📊 Baseline: GPU0={baseline[0]:.0f}MB, GPU1={baseline[1]:.0f}MB, GPU2={baseline[2]:.0f}MB"
        )

        # Build production-like command with all flags
        command = [
            "/llama.cpp/build/bin/llama-server",
            "--host",
            "127.0.0.1",
            "--port",
            "8001",
            "--threads",
            str(config.threads),
            "--ctx-size",
            str(config.context_size),
            "--batch-size",
            str(config.batch_size),
            "--ubatch-size",
            str(config.ubatch_size),
            "--cache-type-k",
            config.cache_type_k,
            "--cache-type-v",
            config.cache_type_v,
            "--numa",
            config.numa,
            "--n-cpu-moe",
            str(config.n_cpu_moe),
            "--gpu-layers",
            str(config.gpu_layers),
            "--split-mode",
            config.split_mode,
            "--tensor-split",
            config.tensor_split,
            "--main-gpu",
            str(config.main_gpu),
            "--model",
            config.gguf_path,
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

        print("🚀 Starting llama-server with production configuration...")

        success = False
        actual_memory = (0.0, 0.0, 0.0)
        error_msg = ""
        oom_detected = False

        try:
            # Start llama-server process
            process = subprocess.Popen(
                ["kubectl", "exec", "-i", "-n", self.namespace, self.pod_name, "--"]
                + command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            print("⏳ Waiting 30 seconds for large model to load...")

            # Wait longer for large models to load
            time.sleep(30)

            # Check if process is still running
            if process.poll() is None:
                # Process is running, get memory
                actual_memory = self._get_actual_memory()
                print(
                    f"📊 Actual: GPU0={actual_memory[0]:.0f}MB, GPU1={actual_memory[1]:.0f}MB, GPU2={actual_memory[2]:.0f}MB"
                )
                success = True
            else:
                # Process exited early, capture output
                stdout, stderr = process.communicate()
                error_output = stderr.strip() if stderr else stdout.strip()

                # Check for OOM patterns
                if any(
                    pattern in error_output.lower()
                    for pattern in [
                        "out of memory",
                        "cuda error",
                        "memory allocation",
                        "cuda out of memory",
                        "insufficient memory",
                        "not enough memory",
                        "failed to allocate",
                        "memory error",
                    ]
                ):
                    oom_detected = True
                    error_msg = f"OOM detected: {error_output[:200]}"
                    print(f"💥 OOM detected: Model requires more than available VRAM")
                else:
                    error_msg = f"Process exited early: {error_output[:200]}"
                    print(f"❌ Process failed: {error_msg}")

                actual_memory = self._get_actual_memory()

            # Terminate the process if still running
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
            # Clean up any remaining processes
            self._cleanup_processes()

            # Verify memory is back to baseline
            cleanup_memory = self._get_actual_memory()
            print(
                f"🔍 Post-cleanup memory: GPU0={cleanup_memory[0]:.0f}MB, GPU1={cleanup_memory[1]:.0f}MB, GPU2={cleanup_memory[2]:.0f}MB"
            )

        # Calculate results
        actual_memory_mb = [actual_memory[i] - baseline[i] for i in range(3)]
        total_actual_gb = sum(actual_memory_mb) / 1024

        accuracy_ratio = total_actual_gb / estimated_gb if estimated_gb > 0 else 0
        exceeds_48gb = self._check_memory_threshold(total_actual_gb) or oom_detected

        result = LargeModelMemoryResult(
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
            exceeds_48gb_vram=exceeds_48gb,
        )

        if success:
            print(
                f"✅ Success: {total_actual_gb:.2f}GB actual vs {estimated_gb:.2f}GB estimated ({accuracy_ratio:.2f}x)"
            )
            if exceeds_48gb:
                print(f"⚠️  Model requires more than 48GB VRAM")
        elif oom_detected:
            print(
                f"💥 OOM: Model requires more than available VRAM ({estimated_gb:.2f}GB estimated)"
            )
        else:
            print(f"❌ Failed: {error_msg}")

        return result

    def collect_large_model_samples(self):
        """Collect real memory samples from large models (30B+, VL models)"""

        print("🎯 Starting Large Model Real Memory Sample Collection")
        print("=" * 80)

        # Define large model test configurations based on production usage patterns
        test_configs = [
            # Qwen3-VL-32B-Thinking tests (production context sizes, high batch)
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=40960,  # 40K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="32B VL baseline 40K context, production batch",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="32B VL medium 128K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=262144,  # 256K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="32B VL large 256K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=524288,  # 512K
                batch_size=1024,
                ubatch_size=1024,
                gpu_layers=-1,
                notes="32B VL huge 512K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=1048576,  # 1M
                batch_size=512,
                ubatch_size=512,
                gpu_layers=-1,
                notes="32B VL extreme 1M context",
            ),
            # Qwen3-30B-A3B tests (text only, high context)
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=40960,  # 40K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B MoE baseline 40K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B MoE medium 128K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=262144,  # 256K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="30B MoE large 256K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=524288,  # 512K
                batch_size=1024,
                ubatch_size=1024,
                gpu_layers=-1,
                notes="30B MoE huge 512K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=1048576,  # 1M
                batch_size=512,
                ubatch_size=512,
                gpu_layers=-1,
                notes="30B MoE extreme 1M context",
            ),
            # Qwen3-Coder-30B-A3B tests (coding context)
            LargeModelTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B Coder baseline 128K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=262144,  # 256K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B Coder medium 256K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=524288,  # 512K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="30B Coder large 512K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-coder-30b-a3b",
                model_name="Qwen3-Coder-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
                context_size=1048576,  # 1M
                batch_size=1024,
                ubatch_size=1024,
                gpu_layers=-1,
                notes="30B Coder extreme 1M context",
            ),
            # Qwen3-VL-30B-A3B tests (VL with thinking)
            LargeModelTestConfiguration(
                model_id="qwen3-vl-30b-a3b-thinking",
                model_name="Qwen3-VL-30B-A3B-Thinking",
                param_size="30B",
                gguf_path="/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf",
                mmproj_path="/models/qwen3-vl-30b-a3b/mmproj.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B VL Thinking baseline 128K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-30b-a3b-thinking",
                model_name="Qwen3-VL-30B-A3B-Thinking",
                param_size="30B",
                gguf_path="/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf",
                mmproj_path="/models/qwen3-vl-30b-a3b/mmproj.gguf",
                context_size=262144,  # 256K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=-1,
                notes="30B VL Thinking medium 256K context",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-30b-a3b-thinking",
                model_name="Qwen3-VL-30B-A3B-Thinking",
                param_size="30B",
                gguf_path="/models/qwen3-vl-30b-a3b/qwen3-vl-30b-a3b-q4-k-m.gguf",
                mmproj_path="/models/qwen3-vl-30b-a3b/mmproj.gguf",
                context_size=524288,  # 512K
                batch_size=2048,
                ubatch_size=2048,
                gpu_layers=-1,
                notes="30B VL Thinking large 512K context",
            ),
            # GPU layer variation tests (30B models with different GPU allocation)
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=40,  # Partial GPU offload
                notes="30B MoE partial GPU layers (40)",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-30b-a3b",
                model_name="Qwen3-30B-A3B",
                param_size="30B",
                gguf_path="/models/qwen3-30b-a3b/qwen3-30b-a3b-iq4-xs-abliterated.gguf",
                context_size=131072,  # 128K
                batch_size=4096,
                ubatch_size=4096,
                gpu_layers=20,  # Lower GPU offload
                notes="30B MoE low GPU layers (20)",
            ),
            # Batch size variation tests (production range 2048-8192)
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=131072,  # 128K
                batch_size=8192,  # High batch
                ubatch_size=8192,
                gpu_layers=-1,
                notes="32B VL high batch size 8192",
            ),
            LargeModelTestConfiguration(
                model_id="qwen3-vl-32b-thinking",
                model_name="Qwen3-VL-32B-Thinking",
                param_size="32B",
                gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
                mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
                context_size=131072,  # 128K
                batch_size=2048,  # Medium batch
                ubatch_size=2048,
                gpu_layers=-1,
                notes="32B VL medium batch size 2048",
            ),
        ]

        total_tests = len(test_configs)
        successful_tests = []
        failed_tests = []
        oom_tests = []

        print(f"📋 Running {total_tests} large model real memory tests")

        for i, config in enumerate(test_configs, 1):
            print(f"\n{'='*60}")
            print(f"🔍 Test {i}/{total_tests} - {config.notes}")

            result = self._run_large_model_test(config)
            self.results.append(result)

            if result.success:
                successful_tests.append(result)
            elif result.oom_detected:
                oom_tests.append(result)
            else:
                failed_tests.append(result)

            # Wait between tests to allow system to stabilize
            if i < total_tests:
                print("⏸️  Waiting 10 seconds between tests...")
                time.sleep(10)

        # Save results
        self._save_large_model_samples()
        self._generate_large_model_summary()

        return len(successful_tests)

    def _save_large_model_samples(self):
        """Save successful large model samples for unit testing"""

        successful_samples = []
        for result in self.results:
            if result.success:
                # Convert to format matching existing samples structure
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
                    # Additional production fields
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
                        "cont_batching": result.config.cont_batching,
                    },
                }
                successful_samples.append(sample)

        # Merge with existing samples
        import os

        existing_samples_path = "/Users/lons7862/workspace/llmmllab/inference/test/unit/real_memory_samples.json"

        if os.path.exists(existing_samples_path):
            with open(existing_samples_path, "r") as f:
                existing_samples = json.load(f)
        else:
            existing_samples = []

        # Combine samples
        all_samples = existing_samples + successful_samples

        # Save combined samples
        with open(existing_samples_path, "w") as f:
            json.dump(all_samples, f, indent=2)

        print(
            f"✅ Merged {len(successful_samples)} large model samples with {len(existing_samples)} existing samples"
        )
        print(f"   Total samples: {len(all_samples)}")

        # Save detailed large model results separately
        large_model_results_path = "/Users/lons7862/workspace/llmmllab/inference/debug/large_model_memory_results.json"
        with open(large_model_results_path, "w") as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)

        print(f"✅ Saved detailed large model results to {large_model_results_path}")

    def _generate_large_model_summary(self):
        """Generate summary of large model results"""

        successful_results = [r for r in self.results if r.success]
        failed_results = [
            r for r in self.results if not r.success and not r.oom_detected
        ]
        oom_results = [r for r in self.results if r.oom_detected]

        print("\n" + "=" * 80)
        print("📊 LARGE MODEL MEMORY SAMPLE COLLECTION RESULTS")
        print("=" * 80)

        print(
            f"\n📈 Summary: {len(successful_results)} successful, {len(failed_results)} failed, {len(oom_results)} OOM"
        )

        if successful_results:
            print(f"\n✅ Successfully Collected Large Model Samples:")
            print(
                "Model                            Context    Batch    Est GB   Act GB   Accuracy"
            )
            print("-" * 85)
            for result in successful_results:
                ctx_str = (
                    f"{result.config.context_size//1024}K"
                    if result.config.context_size >= 1024
                    else str(result.config.context_size)
                )
                mmproj_indicator = "+VL" if result.config.mmproj_path else ""
                print(
                    f"{result.config.model_name[:24] + mmproj_indicator:<28} {ctx_str:>8} {result.config.batch_size:>8} {result.estimated_gb:>8.1f} {result.total_actual_gb:>8.1f} {result.accuracy_ratio:>8.2f}"
                )

        if oom_results:
            print(f"\n💥 OOM Tests (Require >48GB VRAM):")
            for result in oom_results:
                ctx_str = (
                    f"{result.config.context_size//1024}K"
                    if result.config.context_size >= 1024
                    else str(result.config.context_size)
                )
                print(
                    f"- {result.config.model_name} @ {ctx_str}: {result.estimated_gb:.1f}GB estimated"
                )

        if failed_results:
            print(f"\n❌ Failed Tests:")
            for result in failed_results:
                print(
                    f"- {result.config.model_name} @ {result.config.context_size//1024}K: {result.error_msg[:100]}"
                )

        if successful_results:
            accuracy_ratios = [r.accuracy_ratio for r in successful_results]
            print(f"\n📈 Large Model Accuracy Statistics:")
            print(f"Average: {sum(accuracy_ratios)/len(accuracy_ratios):.2f}x")
            print(f"Min: {min(accuracy_ratios):.2f}x")
            print(f"Max: {max(accuracy_ratios):.2f}x")
            print(f"Range: {min(accuracy_ratios):.2f}x - {max(accuracy_ratios):.2f}x")

            # Memory usage statistics
            memory_usages = [r.total_actual_gb for r in successful_results]
            print(f"\nActual Memory Usage:")
            print(f"Min: {min(memory_usages):.1f}GB")
            print(f"Max: {max(memory_usages):.1f}GB")
            print(f"Average: {sum(memory_usages)/len(memory_usages):.1f}GB")


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

    collector = LargeModelSampleCollector(k8s_pod_name=pod_name)
    collector.collect_large_model_samples()
