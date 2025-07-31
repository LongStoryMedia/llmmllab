import os
import torch
import gc
import logging
import time
import subprocess
import signal
import psutil
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import nvsmi

from models.defaults import new_dev_stats
from models.dev_stats import DevStats


class EnhancedHardwareManager:
    """Enhanced service for managing hardware resources with full GPU control."""

    def __init__(self):
        """Initialize the hardware manager and discover available GPU resources."""
        self.has_gpu = torch.cuda.is_available()
        self.logger = logging.getLogger("hardware_manager")

        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Discover all available GPUs
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        self.devices: List[torch.device] = []
        self.current_device_idx = 0
        self.fallback_devices: List[int] = []
        self.blacklisted_devices: List[int] = []

        # Initialize context reset tracking
        self.last_context_reset_time = 0
        self.context_reset_count = 0
        self.last_device_reset = -1

        # Process tracking for GPU cleanup
        self.tracked_processes: Dict[int, Dict] = {}
        self.gpu_process_cache: Dict[int, List] = {}

        if self.has_gpu:
            self.logger.info(f"Detected {self.gpu_count} GPU(s)")
            for i in range(self.gpu_count):
                device = torch.device(f"cuda:{i}")
                self.devices.append(device)
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    self.logger.info(f"  GPU {i}: {gpu_name}")
                except RuntimeError as e:
                    self.logger.warning(f"Could not get name for GPU {i}: {e}")
                    self.logger.info("Attempting to reset CUDA context...")
                    self._reset_cuda_context()
                    try:
                        gpu_name = torch.cuda.get_device_name(i)
                        self.logger.info(
                            f"  GPU {i}: {gpu_name} (after reset)")
                    except:
                        self.logger.error(
                            f"Still unable to get GPU {i} name after reset")

            # Select first device as default
            self.device = self.devices[0]
        else:
            self.device = torch.device("cpu")
            self.logger.warning("No GPU detected, running in CPU mode")

        # Initialize memory stats for each device
        self.memory_stats: Dict[str, DevStats] = {}

        # Apply global PyTorch memory settings
        if self.has_gpu:
            # More aggressive memory management settings
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,"
                "max_split_size_mb:64,"
                "roundup_power2_divisions:2,"
                "garbage_collection_threshold:0.8"
            )

            # Additional CUDA environment variables for better control
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(self.gpu_count))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

            # Log initial state
            self.update_all_memory_stats()
            self.logger.info(
                f"Initialized EnhancedHardwareManager with primary device: {self.device}")
        else:
            self.logger.info("No GPU detected, running in CPU mode")

        # Track OOM occurrences
        self.oom_count = 0
        self.last_oom_time = 0
        self.device_oom_count: Dict[int, int] = {
            i: 0 for i in range(self.gpu_count)} if self.has_gpu else {}

    def _get_gpu_processes(self, device_idx: int) -> List[Dict]:
        """Get all processes currently using a specific GPU."""
        try:
            # Use nvidia-ml-py for more detailed process information
            gpus = nvsmi.get_gpus()
            if device_idx < len(gpus):
                gpu = gpus[device_idx]
                processes = []

                # Try to get process information from nvidia-smi
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                         '--format=csv,noheader,nounits', f'--id={device_idx}'],
                        capture_output=True, text=True, timeout=10
                    )

                    if result.returncode == 0:
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                parts = line.split(', ')
                                if len(parts) >= 3:
                                    pid = int(parts[0])
                                    name = parts[1]
                                    memory_mb = int(parts[2])
                                    processes.append({
                                        'pid': pid,
                                        'name': name,
                                        'memory_mb': memory_mb,
                                        'device_idx': device_idx
                                    })
                except subprocess.TimeoutExpired:
                    self.logger.warning("nvidia-smi query timed out")
                except Exception as e:
                    self.logger.warning(
                        f"Failed to get process info via nvidia-smi: {e}")

                return processes

        except Exception as e:
            self.logger.error(
                f"Error getting GPU processes for device {device_idx}: {e}")

        return []

    def _kill_gpu_processes(self, device_idx: int, exclude_current: bool = True, force: bool = False) -> int:
        """
        Kill processes using a specific GPU.

        Args:
            device_idx: GPU device index
            exclude_current: If True, don't kill the current process
            force: If True, use SIGKILL instead of SIGTERM

        Returns:
            Number of processes killed
        """
        current_pid = os.getpid()
        killed_count = 0

        processes = self._get_gpu_processes(device_idx)

        for proc_info in processes:
            pid = proc_info['pid']

            # Skip current process if requested
            if exclude_current and pid == current_pid:
                continue

            try:
                # Check if process still exists
                if psutil.pid_exists(pid):
                    process = psutil.Process(pid)
                    process_name = process.name()

                    self.logger.warning(
                        f"Terminating GPU process: PID {pid} ({process_name}) "
                        f"using {proc_info['memory_mb']}MB on GPU {device_idx}"
                    )

                    if force:
                        process.kill()  # SIGKILL
                    else:
                        process.terminate()  # SIGTERM

                        # Wait a bit for graceful shutdown
                        try:
                            process.wait(timeout=5)
                        except psutil.TimeoutExpired:
                            # Force kill if it doesn't terminate gracefully
                            self.logger.warning(
                                f"Force killing PID {pid} after timeout")
                            process.kill()

                    killed_count += 1

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                self.logger.debug(f"Could not terminate PID {pid}: {e}")
            except Exception as e:
                self.logger.error(f"Error terminating PID {pid}: {e}")

        if killed_count > 0:
            # Wait a moment for processes to actually exit
            time.sleep(2)

        return killed_count

    def _reset_gpu_driver(self, device_idx: int) -> bool:
        """
        Reset GPU driver for a specific device (requires root/admin privileges).

        WARNING: This is a nuclear option that may affect system stability.
        """
        try:
            self.logger.warning(
                f"Attempting driver reset for GPU {device_idx}")

            # First try nvidia-smi reset
            result = subprocess.run(
                ['nvidia-smi', '--gpu-reset', f'--id={device_idx}'],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                self.logger.info(f"Successfully reset GPU {device_idx} driver")
                # Wait for driver to stabilize
                time.sleep(5)
                return True
            else:
                self.logger.error(f"GPU reset failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.logger.error(f"GPU reset timed out for device {device_idx}")
        except FileNotFoundError:
            self.logger.error("nvidia-smi not found - cannot reset GPU driver")
        except Exception as e:
            self.logger.error(f"Error resetting GPU driver: {e}")

        return False

    def _reset_cuda_context(self, device_idx: Optional[int] = None):
        """Reset the CUDA context to recover from errors."""
        now = time.time()

        # Don't reset too frequently
        if now - self.last_context_reset_time < 30:  # Reduced from 60 to 30 seconds
            self.context_reset_count += 1
            if self.context_reset_count > 5:  # Increased threshold
                self.logger.warning(
                    "Too many context resets in a short time, skipping reset")
                return
        else:
            self.context_reset_count = 0

        self.last_context_reset_time = now

        try:
            # More aggressive context reset
            if device_idx is not None:
                devices_to_reset = [device_idx]
            else:
                devices_to_reset = list(range(self.gpu_count))

            for i in devices_to_reset:
                try:
                    with torch.cuda.device(i):
                        # Clear all cached memory
                        torch.cuda.empty_cache()

                        # Reset memory stats
                        torch.cuda.reset_peak_memory_stats(i)
                        torch.cuda.reset_accumulated_memory_stats(i)

                        # Synchronize to ensure operations complete
                        torch.cuda.synchronize(i)

                        # Create and delete a tensor to reinitialize context
                        temp = torch.ones(1, device=f"cuda:{i}")
                        del temp

                        # Another synchronization
                        torch.cuda.synchronize(i)

                    self.logger.info(f"Reset CUDA context for device {i}")
                    self.last_device_reset = i

                except Exception as e:
                    self.logger.error(f"Failed to reset device {i}: {e}")

            # Global garbage collection
            gc.collect()

            # Try to trigger CUDA context cleanup
            if hasattr(torch.cuda, 'ipc_collect'):
                torch.cuda.ipc_collect()

            self.logger.info("CUDA context reset completed")

        except Exception as e:
            self.logger.error(f"Error during CUDA context reset: {e}")

    def nuclear_clear_memory(self, device_idx: Optional[int] = None, kill_processes: bool = False) -> None:
        """
        Nuclear option: Clear ALL memory from GPU(s), including other processes.

        Args:
            device_idx: Specific device to clear, or None for all devices
            kill_processes: If True, kill other processes using the GPU

        WARNING: This will affect other processes using the GPU!
        """
        if not self.has_gpu:
            self.logger.warning("No GPU available for nuclear memory clearing")
            return

        devices_to_clear = [device_idx] if device_idx is not None else list(
            range(self.gpu_count))

        for dev_idx in devices_to_clear:
            self.logger.warning(
                f"Starting nuclear memory clear for GPU {dev_idx}")

            # Step 1: Kill other processes if requested
            if kill_processes:
                killed = self._kill_gpu_processes(
                    dev_idx, exclude_current=True, force=False)
                if killed > 0:
                    self.logger.warning(
                        f"Killed {killed} processes on GPU {dev_idx}")
                    time.sleep(3)  # Wait for processes to fully exit

                    # Force kill any remaining processes
                    remaining = self._kill_gpu_processes(
                        dev_idx, exclude_current=True, force=True)
                    if remaining > 0:
                        self.logger.warning(
                            f"Force killed {remaining} remaining processes on GPU {dev_idx}")
                        time.sleep(2)

            # Step 2: Clear current process memory aggressively
            try:
                with torch.cuda.device(dev_idx):
                    # Clear all tensors and cache
                    torch.cuda.empty_cache()

                    # Reset all memory tracking
                    torch.cuda.reset_peak_memory_stats(dev_idx)
                    torch.cuda.reset_accumulated_memory_stats(dev_idx)

                    # Synchronize
                    torch.cuda.synchronize(dev_idx)

                    # Try to allocate and free large blocks to trigger cleanup
                    try:
                        props = torch.cuda.get_device_properties(dev_idx)
                        total_memory = props.total_memory

                        # Try to allocate 90% of total memory and immediately free it
                        # Divide by 4 for float32
                        test_size = int(total_memory * 0.9 // 4)
                        test_tensor = torch.empty(
                            test_size, dtype=torch.float32, device=f"cuda:{dev_idx}")
                        del test_tensor

                    except torch.cuda.OutOfMemoryError:
                        # Try smaller allocations
                        for fraction in [0.5, 0.25, 0.1]:
                            try:
                                test_size = int(total_memory * fraction // 4)
                                test_tensor = torch.empty(
                                    test_size, dtype=torch.float32, device=f"cuda:{dev_idx}")
                                del test_tensor
                                break
                            except torch.cuda.OutOfMemoryError:
                                continue

                    # Final cache clear
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(dev_idx)

            except Exception as e:
                self.logger.error(
                    f"Error during memory clearing for GPU {dev_idx}: {e}")

            # Step 3: Reset CUDA context
            self._reset_cuda_context(dev_idx)

            # Step 4: System-level cleanup (if available)
            try:
                # Try to flush GPU memory at driver level
                result = subprocess.run(
                    ['nvidia-smi', '--gpu-reset', f'--id={dev_idx}'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    self.logger.info(
                        f"Successfully reset GPU {dev_idx} at driver level")
                    time.sleep(3)

            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                self.logger.debug(f"Could not reset GPU at driver level: {e}")

        # Global cleanup
        gc.collect()

        # Update stats
        self.update_all_memory_stats()

        for dev_idx in devices_to_clear:
            stats = self.memory_stats.get(str(dev_idx), new_dev_stats())
            self.logger.info(
                f"Nuclear clear complete for GPU {dev_idx}. "
                f"Free memory: {self.format_bytes(stats.mem_free * 1024 * 1024)}"
            )

    def force_set_device(self, device_idx: int) -> None:
        """Force set the current device to a specific GPU index."""
        if not self.has_gpu or device_idx < 0 or device_idx >= self.gpu_count:
            raise ValueError(f"Invalid device index: {device_idx}")

        self.current_device_idx = device_idx
        self.device = self.devices[device_idx]

        try:
            torch.cuda.set_device(device_idx)
        except Exception as e:
            self.logger.warning(f"Error setting CUDA device: {e}")
            self._reset_cuda_context(device_idx)
            try:
                torch.cuda.set_device(device_idx)
            except Exception as e2:
                self.logger.error(
                    f"Failed to set device even after reset: {e2}")

        self.logger.info(
            f"Force set current device to GPU {device_idx}: {self.device}")

    def update_all_memory_stats(self) -> Dict[str, DevStats]:
        """Update memory statistics for all devices."""
        if not self.has_gpu:
            self.memory_stats = {"cpu": new_dev_stats()}
            return self.memory_stats

        try:
            gpus = nvsmi.get_gpus()
            for gpu in gpus:
                self.memory_stats[gpu.id] = self.update_memory_stats(gpu)
        except Exception as e:
            self.logger.error(f"Error updating memory stats: {e}")

        return self.memory_stats

    def format_bytes(self, bytes_value: Union[int, float, str, None]) -> str:
        """Format bytes into human-readable format with appropriate unit."""
        if not isinstance(bytes_value, (int, float)):
            return str(bytes_value)

        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        value = float(bytes_value)

        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024
            unit_index += 1

        return f"{value:.2f} {units[unit_index]}"

    def update_memory_stats(self, gpu: nvsmi.GPU) -> DevStats:
        """Update and return memory statistics for a specific device."""
        device_stats = new_dev_stats()
        if not self.has_gpu:
            self.memory_stats = {"cpu": device_stats}
            return device_stats

        if gpu is not None:
            try:
                device_stats.uuid = gpu.uuid
                device_stats.name = gpu.name
                device_stats.id = gpu.id
                device_stats.driver = gpu.driver
                device_stats.serial = gpu.serial
                device_stats.display_mode = gpu.display_mode
                device_stats.display_active = gpu.display_active
                device_stats.temperature = gpu.temperature
                device_stats.gpu_util = gpu.gpu_util
                device_stats.mem_util = gpu.mem_util
                device_stats.mem_total = gpu.mem_total
                device_stats.mem_used = gpu.mem_used
                device_stats.mem_free = gpu.mem_free

            except RuntimeError as e:
                if "CUDA driver library could not be found" in str(e) or "CUDA error" in str(e):
                    self.logger.error(
                        f"CUDA context error for device {gpu}: {e}")
                    self._reset_cuda_context()
                    try:
                        return self.update_memory_stats(gpu)
                    except Exception as e2:
                        self.logger.error(
                            f"Still failed after context reset: {e2}")
                self.logger.error(
                    f"Error getting memory stats for device {gpu}: {e}")

        return device_stats

    def estimate_largest_free_block(self, device_idx: int) -> int:
        """Estimate the largest contiguous block of free memory."""
        if not self.has_gpu:
            return 0

        try:
            free_memory = torch.cuda.get_device_properties(
                device_idx).total_memory - torch.cuda.memory_allocated(device_idx)

            # More conservative estimate due to fragmentation
            estimated_largest_block = int(free_memory * 0.6)

            if torch.cuda.memory_allocated(device_idx) > 0.7 * torch.cuda.get_device_properties(device_idx).total_memory:
                estimated_largest_block = int(free_memory * 0.4)

            return estimated_largest_block

        except Exception as e:
            self.logger.warning(f"Error estimating largest free block: {e}")
            return 0

    def check_memory_available(self, required_bytes: float) -> bool:
        """Check if there is enough memory available on any GPU for the required amount."""
        if not self.has_gpu:
            self.logger.warning("No GPU available for memory check")
            return False

        for (name, stats) in self.update_all_memory_stats().items():
            try:
                total_mem = stats.mem_total * 1024 * 1024
                free_mem = stats.mem_free * 1024 * 1024
                available_mem = free_mem * 0.8

                self.logger.debug(
                    f"GPU {name}: Total: {total_mem/1e9:.2f}GB, "
                    f"Free: {free_mem/1e9:.2f}GB, Required: {required_bytes/1e9:.2f}GB"
                )

                if available_mem >= required_bytes:
                    return True

            except Exception as e:
                self.logger.error(f"Error checking memory on GPU {name}: {e}")
                continue

        self.logger.warning(
            f"Not enough memory available. Required: {required_bytes/1e9:.2f}GB")
        return False

    def clear_memory(self, device_idx: Optional[int] = None, aggressive: bool = False, nuclear: bool = False) -> None:
        """
        Clear unused memory and run garbage collection.

        Args:
            device_idx: Optional specific device to clear, otherwise clear all
            aggressive: If True, perform more aggressive memory clearing
            nuclear: If True, perform nuclear memory clearing (affects other processes)
        """
        if not self.has_gpu:
            return

        if nuclear:
            self.nuclear_clear_memory(device_idx, kill_processes=True)
            return

        # Standard garbage collection
        gc.collect()

        if device_idx is not None:
            if aggressive:
                # More aggressive clearing for specific device
                try:
                    with torch.cuda.device(device_idx):
                        # Clear cache multiple times
                        for _ in range(3):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(device_idx)

                        # Try defragmentation technique
                        free_mem = torch.cuda.get_device_properties(
                            device_idx).total_memory - torch.cuda.memory_allocated(device_idx)

                        if free_mem > 1024 * 1024 * 512:  # At least 512MB free
                            try:
                                # Allocate and free large blocks to reduce fragmentation
                                test_size = int(free_mem * 0.3)
                                test_tensor = torch.empty(
                                    test_size // 4, dtype=torch.float32, device=f"cuda:{device_idx}")
                                del test_tensor
                                torch.cuda.empty_cache()
                            except torch.cuda.OutOfMemoryError:
                                pass

                except Exception as e:
                    self.logger.warning(
                        f"Error during aggressive memory clearing: {e}")

            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
        else:
            if aggressive:
                for i in range(self.gpu_count):
                    self.clear_memory(i, aggressive=True)
            else:
                torch.cuda.empty_cache()

        self.update_all_memory_stats()

    def get_gpu_process_info(self, device_idx: Optional[int] = None) -> Dict[int, List[Dict]]:
        """Get information about all processes using GPUs."""
        if not self.has_gpu:
            return {}

        devices_to_check = [device_idx] if device_idx is not None else list(
            range(self.gpu_count))
        process_info = {}

        for dev_idx in devices_to_check:
            process_info[dev_idx] = self._get_gpu_processes(dev_idx)

        return process_info

    def monitor_gpu_usage(self, duration_seconds: int = 60, interval_seconds: int = 5) -> None:
        """Monitor GPU usage over time and log statistics."""
        if not self.has_gpu:
            self.logger.info("No GPU available for monitoring")
            return

        self.logger.info(
            f"Starting GPU monitoring for {duration_seconds} seconds (interval: {interval_seconds}s)")

        start_time = time.time()

        while time.time() - start_time < duration_seconds:
            self.update_all_memory_stats()

            for dev_idx in range(self.gpu_count):
                stats = self.memory_stats.get(str(dev_idx), new_dev_stats())
                processes = self._get_gpu_processes(dev_idx)

                self.logger.info(
                    f"GPU {dev_idx}: {stats.mem_util}% memory, {stats.gpu_util}% compute, "
                    f"{len(processes)} processes, {stats.temperature}°C"
                )

            time.sleep(interval_seconds)


# Create a singleton instance
hardware_manager = EnhancedHardwareManager()
