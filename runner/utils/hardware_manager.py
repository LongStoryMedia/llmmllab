import os
import torch
import gc
import logging
import time
import subprocess
import psutil
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import nvsmi
from runner.models.dev_stats import DevStats


def is_memory_related_error(error: Union[str, Exception]) -> bool:
    """Check if an error is memory-related."""
    error_str = str(error).lower()
    keywords = [
        "out of memory",
        "memory allocation",
        "cuda error",
        "insufficient memory",
        "allocation failed",
        "oom",
    ]
    return any(keyword in error_str for keyword in keywords)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""

    safety_margin: float = 0.8
    defrag_threshold: float = 0.9
    critical_threshold: float = 0.95
    context_reset_cooldown: int = 30

    # Thermal / power management
    gpu_power_cap_pct: float = float(os.getenv("GPU_POWER_CAP_PCT", "85"))
    thermal_warning_c: float = 78.0   # Log warning above this
    thermal_critical_c: float = 88.0  # Considered critical — risk of PCIe drop


class CUDAContextManager:
    """Handles CUDA context operations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.last_reset_time = 0
        self.reset_count = 0
        self.context_initialized: Dict[int, bool] = {}

    def destroy_context(self, device_idx: int) -> bool:
        """
        Completely destroy CUDA context to release all GPU memory.
        This is more aggressive than reset and actually removes the context.
        """
        try:
            # Method 1: Try ctypes to call cudaDeviceReset
            if self._try_cuda_device_reset(device_idx):
                self.context_initialized[device_idx] = False
                self.logger.info(
                    f"Successfully destroyed CUDA context for device {device_idx}"
                )
                return True

            # Method 2: Fallback to aggressive PyTorch reset
            self.logger.warning(
                f"Using fallback context destruction for device {device_idx}"
            )
            self._pytorch_aggressive_reset(device_idx)
            self.context_initialized[device_idx] = False
            return True

        except Exception as e:
            self.logger.error(
                f"Context destruction failed for device {device_idx}: {e}"
            )
            return False

    def _try_cuda_device_reset(self, device_idx: int) -> bool:
        """Try to call cudaDeviceReset via ctypes."""
        try:
            import ctypes

            # Try multiple library names
            cuda = None
            for lib_name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
                try:
                    cuda = ctypes.CDLL(lib_name)
                    break
                except OSError:
                    continue

            if cuda is None:
                return False

            # Set device and reset
            try:
                if hasattr(cuda, "cudaSetDevice"):
                    cuda.cudaSetDevice(device_idx)

                if hasattr(cuda, "cudaDeviceReset"):
                    result = cuda.cudaDeviceReset()
                    if result == 0:
                        self.logger.info(
                            f"cudaDeviceReset successful for device {device_idx}"
                        )
                        return True
            except Exception as e:
                self.logger.debug(f"cudaDeviceReset failed: {e}")

            return False

        except ImportError:
            return False

    def reset_context(self, device_idx: int, cooldown: int = 30) -> bool:
        """Reset CUDA context with cooldown protection."""
        now = time.time()

        # Cooldown check
        if now - self.last_reset_time < cooldown:
            self.reset_count += 1
            if self.reset_count > 5:
                self.logger.warning("Too many resets, skipping")
                return False
        else:
            self.reset_count = 0

        self.last_reset_time = now

        try:
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device_idx)
                torch.cuda.reset_accumulated_memory_stats(device_idx)
                torch.cuda.synchronize(device_idx)

            # Try advanced reset methods
            if self._try_pynvml_reset(device_idx):
                return True

            # Fallback: aggressive PyTorch cleanup
            self._pytorch_aggressive_reset(device_idx)
            return True

        except Exception as e:
            self.logger.error(f"Context reset failed for device {device_idx}: {e}")
            return False

    def _try_pynvml_reset(self, device_idx: int) -> bool:
        """Try pynvml-based reset."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)

            # Try available reset methods
            try:
                pynvml.nvmlDeviceResetApplicationsClocks(handle)
                self.logger.debug(f"pynvml reset successful for device {device_idx}")
                return True
            except:
                pass

        except Exception as e:
            self.logger.debug(f"pynvml reset not available: {e}")
        return False

    def _pytorch_aggressive_reset(self, device_idx: int):
        """Aggressive PyTorch memory reset."""
        with torch.cuda.device(device_idx):
            # Multiple cache clears
            for _ in range(3):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device_idx)

            # Defragmentation through allocation
            try:
                props = torch.cuda.get_device_properties(device_idx)
                total_mem = props.total_memory

                for fraction in [0.5, 0.25, 0.1]:
                    try:
                        size = int(total_mem * fraction // 4)
                        temp = torch.empty(
                            size, dtype=torch.float32, device=f"cuda:{device_idx}"
                        )
                        del temp
                        torch.cuda.empty_cache()
                        break
                    except torch.cuda.OutOfMemoryError:
                        continue
            except Exception as e:
                self.logger.debug(f"Defragmentation failed: {e}")


class GPUProcessManager:
    """Manages GPU process operations."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def get_processes(self, device_idx: int) -> List[Dict]:
        """Get processes using a specific GPU."""
        try:
            cmd = [
                "nvidia-smi",
                f"--id={device_idx}",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, check=False
            )

            processes = []
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        pid, name, memory = line.split(",")
                        processes.append(
                            {
                                "pid": int(pid),
                                "name": name.strip(),
                                "memory_mb": int(memory.strip()),
                                "device_idx": device_idx,
                            }
                        )
            return processes

        except Exception as e:
            self.logger.error(f"Error getting GPU processes: {e}")
            return []

    def kill_process(self, pid: int) -> bool:
        """Kill a specific process."""
        try:
            if not psutil.pid_exists(pid):
                return False

            process = psutil.Process(pid)
            self.logger.info(f"Terminating PID {pid} ({process.name()})")
            process.terminate()

            # try:
            #     process.wait(timeout=10)
            # except psutil.TimeoutExpired:
            #     self.logger.warning(f"Force killing PID {pid}")
            #     process.kill()

            return True

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.debug(f"Could not terminate PID {pid}: {e}")
            return False

    def kill_gpu_processes(
        self, device_idx: int, exclude_current: bool = True, pid: Optional[int] = None
    ) -> int:
        """
        Kill processes on a GPU.

        Args:
            device_idx: GPU device index
            exclude_current: If True, don't kill the current process
            pid: If specified, only kill this specific PID

        Returns:
            Number of processes killed
        """
        current_pid = os.getpid()
        killed = 0

        # If specific PID requested
        if pid is not None:
            if exclude_current and pid == current_pid:
                self.logger.debug(f"Skipping current process PID {pid}")
                return 0

            # Check if PID is on this GPU
            processes = self.get_processes(device_idx)
            if any(proc["pid"] == pid for proc in processes):
                if self.kill_process(pid):
                    killed = 1
            else:
                self.logger.debug(f"PID {pid} not found on GPU {device_idx}")

            return killed

        # Kill all processes on GPU
        for proc in self.get_processes(device_idx):
            if exclude_current and proc["pid"] == current_pid:
                continue
            if self.kill_process(proc["pid"]):
                killed += 1

        if killed > 0:
            time.sleep(2)  # Wait for cleanup

        return killed


class MemoryManager:
    """Handles GPU memory operations."""

    def __init__(self, logger: logging.Logger, config: MemoryConfig):
        self.logger = logger
        self.config = config
        self.context_manager = CUDAContextManager(logger)
        self.process_manager = GPUProcessManager(logger)

    def clear_memory(self, device_idx: int, pid: Optional[int] = None):
        """
        Clear GPU memory at different levels.

        Args:
            device_idx: GPU device index
            pid: If specified, only kill this specific PID
        """
        current_pid = os.getpid()
        self.logger.warning(f"Nuclear clear for GPU {device_idx}")

        # Step 1: If targeting current process, destroy our own CUDA context first
        if pid == current_pid or pid is None:
            self.logger.warning(
                f"Destroying CUDA context for current process on GPU {device_idx}"
            )
            self.context_manager.destroy_context(device_idx)

        # Step 2: Kill other processes (or specific PID if not current)
        if pid != current_pid:
            killed = self.process_manager.kill_gpu_processes(
                device_idx, pid=pid, exclude_current=(pid is None)
            )
            if killed > 0:
                self.logger.info(f"Killed {killed} process(es) on GPU {device_idx}")

    def _is_critically_low(self, device_idx: int) -> bool:
        """Check if memory is critically low."""
        try:
            props = torch.cuda.get_device_properties(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            return (allocated / props.total_memory) > self.config.critical_threshold
        except:
            return False

    def _can_defragment(self, device_idx: int) -> bool:
        """Check if it's safe to defragment."""
        try:
            props = torch.cuda.get_device_properties(device_idx)
            allocated = torch.cuda.memory_allocated(device_idx)
            free = props.total_memory - allocated
            return (
                free > (1024 * 1024 * 1024)
                and (allocated / props.total_memory) < self.config.defrag_threshold
            )
        except:
            return False

    def _defragment_memory(self, device_idx: int):
        """Attempt memory defragmentation."""
        try:
            props = torch.cuda.get_device_properties(device_idx)
            free = props.total_memory - torch.cuda.memory_allocated(device_idx)
            size = max(1024 * 1024, int(free * 0.2))

            temp = torch.empty(
                size // 4, dtype=torch.float32, device=f"cuda:{device_idx}"
            )
            del temp
            torch.cuda.empty_cache()
            self.logger.debug(f"Defragmented GPU {device_idx}")

        except Exception as e:
            self.logger.debug(f"Defragmentation failed: {e}")


class EnhancedHardwareManager:
    """Simplified hardware manager with clear separation of concerns."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.logger = self._setup_logging()
        self.config = config or MemoryConfig()

        # GPU detection
        self.has_gpu = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        self.devices: List[torch.device] = []
        self.current_device_idx = 0

        # Managers
        if self.has_gpu:
            self.memory_manager = MemoryManager(self.logger, self.config)
            self.process_manager = GPUProcessManager(self.logger)
            self._init_gpus()
        else:
            self.device = torch.device("cpu")
            self.logger.warning("No GPU detected, running in CPU mode")

        # Memory stats
        self.memory_stats: Dict[str, DevStats] = {}
        self.update_all_memory_stats()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("hardware_manager")
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        return logger

    def _init_gpus(self):
        """Initialize GPU devices."""
        self.logger.info(f"Detected {self.gpu_count} GPU(s)")

        for i in range(self.gpu_count):
            device = torch.device(f"cuda:{i}")
            self.devices.append(device)

            try:
                gpu_name = torch.cuda.get_device_name(i)
                self.logger.info(f"  GPU {i}: {gpu_name}")
            except RuntimeError as e:
                self.logger.warning(f"Could not get name for GPU {i}, resetting: {e}")
                self.memory_manager.context_manager.reset_context(i)

        self.device = self.devices[0]

        # Set environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,max_split_size_mb:64"
        )
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        # Apply power management settings to reduce thermal stress
        self._apply_gpu_power_management()

    # ------------------------------------------------------------------
    # GPU thermal / power management
    # ------------------------------------------------------------------

    def _apply_gpu_power_management(self) -> None:
        """Apply persistence mode and power cap to all GPUs on startup.

        Power-capping consumer GPUs (especially the RTX 3090 at 350 W TDP)
        significantly reduces peak junction temperature with only a minor
        throughput impact on inference workloads which are memory-bandwidth
        bound, not compute bound.

        See docs/gpu_configuration.md § Troubleshooting for background.
        """
        cap_pct = self.config.gpu_power_cap_pct
        if cap_pct <= 0 or cap_pct > 100:
            self.logger.info("GPU power cap disabled (GPU_POWER_CAP_PCT=0 or >100)")
            return

        for i in range(self.gpu_count):
            try:
                # Enable persistence mode (keeps driver loaded, avoids init spikes)
                subprocess.run(
                    ["nvidia-smi", "-i", str(i), "-pm", "1"],
                    capture_output=True, text=True, timeout=10, check=False,
                )

                # Query default power limit
                result = subprocess.run(
                    [
                        "nvidia-smi", "-i", str(i),
                        "--query-gpu=power.default_limit",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=10, check=False,
                )

                if result.returncode != 0 or not result.stdout.strip():
                    self.logger.debug(f"GPU {i}: could not query default power limit")
                    continue

                default_watts = float(result.stdout.strip())
                target_watts = int(default_watts * cap_pct / 100)

                set_result = subprocess.run(
                    ["nvidia-smi", "-i", str(i), "-pl", str(target_watts)],
                    capture_output=True, text=True, timeout=10, check=False,
                )

                if set_result.returncode == 0:
                    self.logger.info(
                        f"GPU {i}: power cap set to {target_watts}W "
                        f"({cap_pct:.0f}% of {default_watts:.0f}W default)"
                    )
                else:
                    # Non-fatal: insufficient permissions or unsupported GPU
                    self.logger.debug(
                        f"GPU {i}: could not set power cap — "
                        f"{set_result.stderr.strip()}"
                    )
            except Exception as e:
                self.logger.debug(f"GPU {i}: power management setup failed: {e}")

    def check_gpu_thermals(self) -> Dict[int, float]:
        """Check GPU temperatures and log warnings for hot devices.

        Returns dict mapping device index → temperature in Celsius.
        Can be called periodically (e.g. from a maintenance loop).
        """
        temps: Dict[int, float] = {}
        if not self.has_gpu:
            return temps

        try:
            self.update_all_memory_stats()
            for dev_id, stats in self.memory_stats.items():
                if not dev_id.isdigit():
                    continue
                idx = int(dev_id)
                temp = stats.temperature or 0.0
                temps[idx] = temp

                if temp >= self.config.thermal_critical_c:
                    self.logger.error(
                        f"🔥 GPU {idx} ({stats.name}) CRITICAL temperature: "
                        f"{temp}°C — risk of PCIe bus drop! "
                        f"Consider reducing workload or improving cooling."
                    )
                elif temp >= self.config.thermal_warning_c:
                    self.logger.warning(
                        f"🌡️ GPU {idx} ({stats.name}) high temperature: {temp}°C"
                    )
        except Exception as e:
            self.logger.debug(f"Thermal check failed: {e}")

        return temps

    def reinitialize_cuda(self, device_idx: int):
        """
        Reinitialize CUDA context after destruction.
        Call this after nuclear clear to be able to use the GPU again.
        """
        try:
            with torch.cuda.device(device_idx):
                # Force context creation
                _ = torch.cuda.current_device()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device_idx)

            self.memory_manager.context_manager.context_initialized[device_idx] = True
            self.logger.info(f"CUDA context reinitialized for device {device_idx}")
            return True

        except Exception as e:
            self.logger.error(
                f"Failed to reinitialize CUDA for device {device_idx}: {e}"
            )
            return False

    def clear_memory(self, device_idx: Optional[int] = None, pid: Optional[int] = None):
        """
        Clear GPU memory with specified strategy.

        Args:
            device_idx: GPU device index (None for all GPUs)
            aggressive: Use aggressive clearing strategy
            nuclear: Use nuclear strategy (destroys CUDA context)
            pid: If specified with nuclear=True, only kill this specific PID
            reinit: If True, reinitialize CUDA context after nuclear clear
        """
        if not self.has_gpu:
            return

        devices = (
            [device_idx] if device_idx is not None else list(range(self.gpu_count))
        )

        for dev_idx in devices:
            self.memory_manager.clear_memory(dev_idx, pid=pid)

            # Reinitialize CUDA context if nuclear and reinit requested
            # if nuclear and reinit:
            #     time.sleep(1)  # Brief pause before reinit
            #     self.reinitialize_cuda(dev_idx)

        self.update_all_memory_stats()

    def check_memory_available(self, required_bytes: float) -> bool:
        """Check if required memory is available across all GPUs."""
        if not self.has_gpu:
            return False

        # Always refresh memory stats before checking to avoid stale data
        self.update_all_memory_stats()

        total_available = 0
        for stats in self.memory_stats.values():
            if stats.mem_free:
                available = stats.mem_free * 1024 * 1024 * self.config.safety_margin
                total_available += available

        self.logger.debug(
            f"Memory check: {total_available/1e9:.2f}GB available vs {required_bytes/1e9:.2f}GB required"
        )
        return total_available >= required_bytes

    def update_all_memory_stats(self) -> Dict[str, DevStats]:
        """Update memory statistics for all devices."""
        if not self.has_gpu:
            return {"cpu": self._empty_dev_stats("cpu")}

        try:
            for gpu in nvsmi.get_gpus():
                self.memory_stats[str(gpu.id)] = self._gpu_to_dev_stats(gpu)
        except Exception as e:
            self.logger.error(f"Error updating memory stats: {e}")

        return self.memory_stats

    def _gpu_to_dev_stats(self, gpu: nvsmi.GPU) -> DevStats:
        """Convert GPU object to DevStats."""
        return DevStats(
            id=gpu.id,
            name=gpu.name,
            uuid=gpu.uuid,
            driver=gpu.driver,
            serial=gpu.serial,
            display_mode=gpu.display_mode,
            display_active=gpu.display_active,
            temperature=gpu.temperature,
            gpu_util=gpu.gpu_util,
            mem_util=gpu.mem_util,
            mem_total=gpu.mem_total,
            mem_used=gpu.mem_used,
            mem_free=gpu.mem_free,
        )

    def _empty_dev_stats(self, device_id: str) -> DevStats:
        """Create empty DevStats."""
        return DevStats(
            id=device_id,
            name="CPU",
            uuid="",
            driver="",
            serial="",
            display_mode="",
            display_active="",
            temperature=0.0,
            gpu_util=0.0,
            mem_util=0.0,
            mem_total=0.0,
            mem_used=0.0,
            mem_free=0.0,
        )

    def get_device_mappings(self) -> Dict[str, Dict[str, Union[int, str]]]:
        """Get mapping of device indices to names."""
        if not self.has_gpu:
            return {"cpu": {"index": -1, "name": "CPU", "uuid": "", "id": "cpu"}}

        mappings = {}
        stats = self.update_all_memory_stats()

        for i in range(self.gpu_count):
            device_id = str(i)
            if device_id in stats:
                s = stats[device_id]
                mappings[device_id] = {
                    "index": i,
                    "name": str(s.name) if s.name else f"GPU {i}",
                    "uuid": str(s.uuid) if s.uuid else "",
                    "id": str(s.id) if s.id else device_id,
                }
            else:
                mappings[device_id] = {
                    "index": i,
                    "name": f"GPU {i}",
                    "uuid": "",
                    "id": device_id,
                }

        mappings["cpu"] = {"index": -1, "name": "CPU", "uuid": "", "id": "cpu"}
        return mappings

    def get_gpu_process_info(
        self, device_idx: Optional[int] = None
    ) -> Dict[int, List[Dict]]:
        """Get process information for GPUs."""
        if not self.has_gpu:
            return {}

        devices = (
            [device_idx] if device_idx is not None else list(range(self.gpu_count))
        )
        return {i: self.process_manager.get_processes(i) for i in devices}

    @staticmethod
    def format_bytes(bytes_value: Union[int, float, str, None]) -> str:
        """Format bytes into human-readable format."""
        if not isinstance(bytes_value, (int, float)):
            return str(bytes_value)

        units = ["B", "KB", "MB", "GB", "TB"]
        value = float(bytes_value)

        for unit in units:
            if value < 1024:
                return f"{value:.2f} {unit}"
            value /= 1024

        return f"{value:.2f} TB"


# Singleton instance
hardware_manager = EnhancedHardwareManager()
