#!/usr/bin/env python3
"""
Conservative Large Model Memory Collection

Collects a few more large model samples with careful resource management
to avoid container crashes. Focuses on models that previously failed.
"""

import subprocess
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ConservativeTestConfig:
    """Conservative test configuration to avoid OOM."""

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


def get_pod_name() -> Optional[str]:
    """Get the ollama pod name."""
    try:
        result = subprocess.run(
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
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        else:
            logger.error(f"Failed to get pod name: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Error getting pod name: {e}")
        return None


def check_container_health() -> bool:
    """Check if the container is healthy and responsive."""
    try:
        pod_name = get_pod_name()
        if not pod_name:
            logger.error("Could not get pod name")
            return False

        result = subprocess.run(
            ["kubectl", "exec", "-n", "ollama", pod_name, "--", "echo", "health_check"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        return result.returncode == 0 and "health_check" in result.stdout
    except Exception as e:
        logger.error(f"Container health check failed: {e}")
        return False


def wait_for_container_recovery(max_wait: int = 60) -> bool:
    """Wait for container to recover after potential crash."""
    logger.info("Waiting for container recovery...")

    for i in range(max_wait):
        if check_container_health():
            logger.info(f"Container healthy after {i} seconds")
            time.sleep(5)  # Additional safety buffer
            return True

        time.sleep(1)
        if i % 10 == 0:
            logger.info(f"Still waiting for container... ({i}/{max_wait}s)")

    logger.error(f"Container did not recover after {max_wait} seconds")
    return False


def get_gpu_memory_usage() -> Tuple[bool, Dict[str, float]]:
    """Get current GPU memory usage safely."""
    try:
        pod_name = get_pod_name()
        if not pod_name:
            logger.error("Could not get pod name for memory check")
            return False, {}

        # Get GPU memory usage
        result = subprocess.run(
            [
                "kubectl",
                "exec",
                "-n",
                "ollama",
                pod_name,
                "--",
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode != 0:
            logger.error(f"nvidia-smi failed: {result.stderr}")
            return False, {}

        memory_values = result.stdout.strip().split("\n")
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


def estimate_memory_conservative(config: ConservativeTestConfig) -> float:
    """Conservative memory estimation to avoid overloading."""
    # Base model size estimates (conservative)
    base_sizes = {
        "32B": 18.0,  # Conservative 18GB for 32B models
        "30B": 16.0,  # Conservative 16GB for 30B models
    }

    # Get base size
    base_gb = base_sizes.get(config.param_size, 10.0)

    # Add context overhead (conservative KV cache estimate)
    context_overhead = (config.context_size / 4096) * 0.5  # 0.5GB per 4K context

    # Add CLIP model if vision
    clip_overhead = 1.2 if config.mmproj_path else 0.0

    # Add safety buffer
    safety_buffer = 2.0

    total = base_gb + context_overhead + clip_overhead + safety_buffer

    logger.info(
        f"Conservative estimate for {config.model_name} @ {config.context_size//1024}K: {total:.1f}GB"
    )
    return total


def run_conservative_test(config: ConservativeTestConfig) -> Optional[Dict]:
    """Run a single conservative test with extensive safety checks."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {config.model_name}")
    logger.info(f"Context: {config.context_size//1024}K, Batch: {config.batch_size}")
    logger.info(f"{'='*60}")

    # Pre-test container health check
    if not check_container_health():
        logger.error("Container not healthy before test")
        if not wait_for_container_recovery():
            return None

    # Estimate memory and check if it's safe
    estimated_gb = estimate_memory_conservative(config)
    if estimated_gb > 28:  # Conservative limit
        logger.warning(f"Estimated memory {estimated_gb:.1f}GB too high, skipping")
        return None

    # Get baseline GPU memory
    baseline_success, baseline_memory = get_gpu_memory_usage()
    if not baseline_success:
        logger.error("Failed to get baseline memory, skipping test")
        return None

    baseline_total = sum(baseline_memory.values())
    logger.info(f"Baseline GPU memory: {baseline_total:.0f}MB")

    try:
        # Get pod name
        pod_name = get_pod_name()
        if not pod_name:
            logger.error("Failed to get pod name")
            return None

        # Build conservative llama-server command
        cmd = [
            "kubectl",
            "exec",
            "-i",
            "-n",
            "ollama",
            pod_name,
            "--",
            "llama-server",
            "--model",
            config.gguf_path,
            "--ctx-size",
            str(config.context_size),
            "--batch-size",
            str(config.batch_size),
            "--ubatch-size",
            str(config.ubatch_size),
            "--gpu-layers",
            str(config.gpu_layers),
            "--threads",
            "16",  # Reduced threads
            "--no-kv-offload",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--numa",
            "distribute",
            "--split-mode",
            "layer",
            "--tensor-split",
            "0.25,0.45,0.30",  # More conservative split
            "--main-gpu",
            "1",
            "--port",
            "8080",
            "--timeout",
            "30",  # Shorter timeout
        ]

        # Add mmproj if vision model
        if config.mmproj_path:
            cmd.extend(["--mmproj", config.mmproj_path])

        logger.info("Starting llama-server with conservative settings...")

        # Start server with timeout
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for startup with shorter timeout
        startup_time = 0
        max_startup = 45  # Reduced startup time

        while startup_time < max_startup:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                logger.error(f"Server crashed during startup: {stderr}")
                return None

            time.sleep(1)
            startup_time += 1

            if startup_time % 10 == 0:
                logger.info(f"Startup progress: {startup_time}/{max_startup}s")

        # Give it a moment to fully initialize
        time.sleep(5)

        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error(f"Server died after startup: {stderr}")
            return None

        # Get memory usage
        success, memory_usage = get_gpu_memory_usage()
        if not success:
            logger.error("Failed to get memory usage")
            process.terminate()
            return None

        logger.info("Memory measurement successful!")
        for gpu, usage in memory_usage.items():
            logger.info(f"  {gpu.upper()}: {usage:.0f}MB")

        total_actual_mb = sum(memory_usage.values())
        total_actual_gb = total_actual_mb / 1024

        logger.info(f"Total GPU memory: {total_actual_gb:.2f}GB")

        # Calculate accuracy
        accuracy_ratio = estimated_gb / total_actual_gb if total_actual_gb > 0 else 0

        # Create result
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
            "conservative_test": True,
            "production_config": {
                "threads": 16,
                "no_kv_offload": True,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "numa": "distribute",
                "split_mode": "layer",
                "tensor_split": "0.25,0.45,0.30",
                "main_gpu": 1,
                "timeout": 30,
            },
        }

        logger.info(f"✅ Success: {config.model_name} @ {config.context_size//1024}K")
        logger.info(
            f"   Estimated: {estimated_gb:.1f}GB, Actual: {total_actual_gb:.1f}GB ({accuracy_ratio:.2f}x)"
        )

        return result

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None

    finally:
        # Always try to stop the server
        try:
            process.terminate()
            process.wait(timeout=10)
        except:
            try:
                process.kill()
            except:
                pass

        # Wait before next test
        logger.info("Cooling down before next test...")
        time.sleep(10)


def main():
    """Run conservative large model memory collection."""

    logger.info("Starting Conservative Large Model Memory Collection")
    logger.info("Focus: Models that previously crashed with safer configurations")

    # Conservative test configurations - focus on models that failed
    test_configs = [
        # Qwen3-VL-32B-Thinking with smaller context (previous 512K crashed)
        ConservativeTestConfig(
            model_id="qwen3-vl-32b-thinking",
            model_name="Qwen3-VL-32B-Thinking",
            param_size="32B",
            gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
            context_size=65536,  # 64K instead of 512K
            batch_size=2048,  # Smaller batch
            ubatch_size=2048,
            gpu_layers=-1,
            mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
            notes="32B VL thinking conservative 64K",
        ),
        # Qwen3-VL-32B-Thinking with even smaller context
        ConservativeTestConfig(
            model_id="qwen3-vl-32b-thinking",
            model_name="Qwen3-VL-32B-Thinking",
            param_size="32B",
            gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
            context_size=131072,  # 128K
            batch_size=1024,  # Even smaller batch
            ubatch_size=1024,
            gpu_layers=-1,
            mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
            notes="32B VL thinking conservative 128K",
        ),
        # Qwen3-Coder-30B-A3B with smaller context (previous 256K crashed)
        ConservativeTestConfig(
            model_id="qwen3-coder-30b-a3b",
            model_name="Qwen3-Coder-30B-A3B",
            param_size="30B",
            gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
            context_size=65536,  # 64K instead of 256K
            batch_size=2048,  # Smaller batch
            ubatch_size=2048,
            gpu_layers=-1,
            mmproj_path=None,
            notes="30B Coder conservative 64K",
        ),
        # Qwen3-Coder-30B-A3B with medium context
        ConservativeTestConfig(
            model_id="qwen3-coder-30b-a3b",
            model_name="Qwen3-Coder-30B-A3B",
            param_size="30B",
            gguf_path="/models/qwen3-coder-30b-a3b/Qwen3-Coder-30B-A3B-Instruct-1M-UD-Q4_K_XL.gguf",
            context_size=131072,  # 128K
            batch_size=1024,  # Small batch
            ubatch_size=1024,
            gpu_layers=-1,
            mmproj_path=None,
            notes="30B Coder conservative 128K",
        ),
        # One more 32B VL variation with minimal resources
        ConservativeTestConfig(
            model_id="qwen3-vl-32b-thinking",
            model_name="Qwen3-VL-32B-Thinking",
            param_size="32B",
            gguf_path="/models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf",
            context_size=32768,  # Just 32K
            batch_size=1024,  # Small batch
            ubatch_size=1024,
            gpu_layers=-1,
            mmproj_path="/models/qwen3-vl-32b/mmproj-bf16.gguf",
            notes="32B VL thinking minimal 32K",
        ),
    ]

    results = []
    successful_tests = 0
    failed_tests = 0

    logger.info(f"Planning to test {len(test_configs)} conservative configurations")

    for i, config in enumerate(test_configs, 1):
        logger.info(f"\n--- Test {i}/{len(test_configs)} ---")

        # Extra container health check before each test
        if not check_container_health():
            logger.warning("Container not healthy, waiting for recovery...")
            if not wait_for_container_recovery(90):
                logger.error("Container not recovered, skipping remaining tests")
                break

        result = run_conservative_test(config)

        if result:
            results.append(result)
            successful_tests += 1
            logger.info(f"✅ Test {i} successful")
        else:
            failed_tests += 1
            logger.error(f"❌ Test {i} failed")

            # After failure, wait longer for recovery
            logger.info("Waiting for recovery after failure...")
            time.sleep(15)

            if not wait_for_container_recovery(120):
                logger.error("Container not recovered after failure, stopping tests")
                break

    # Save results
    if results:
        output_file = "conservative_large_model_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"CONSERVATIVE COLLECTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Successful tests: {successful_tests}")
        logger.info(f"Failed tests: {failed_tests}")
        logger.info(f"Results saved to: {output_file}")

        logger.info(f"\nSuccessful measurements:")
        for result in results:
            logger.info(
                f"  {result['model_name']} @ {result['context_size']//1024}K: "
                f"{result['total_actual_gb']:.1f}GB actual "
                f"(est: {result['estimated_gb']:.1f}GB, {result['accuracy_ratio']:.2f}x)"
            )
    else:
        logger.error("No successful tests completed!")


if __name__ == "__main__":
    main()
