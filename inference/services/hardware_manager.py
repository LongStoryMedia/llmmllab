import os
import torch
import gc
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import nvsmi

from models.defaults import new_dev_stats
from models.dev_stats import DevStats


class HardwareManager:
    """Service for managing hardware resources, particularly GPU memory."""

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
            # Configure PyTorch memory allocation for better fragmentation handling
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
            # Log initial state
            self.update_all_memory_stats()
            self.logger.info(
                f"Initialized HardwareManager with primary device: {self.device}")
        else:
            self.logger.info("No GPU detected, running in CPU mode")

        # Track OOM occurrences
        self.oom_count = 0
        self.last_oom_time = 0
        self.device_oom_count: Dict[int, int] = {
            i: 0 for i in range(self.gpu_count)} if self.has_gpu else {}

    def _reset_cuda_context(self):
        """Reset the CUDA context to recover from errors."""
        now = time.time()

        # Don't reset too frequently
        if now - self.last_context_reset_time < 60:  # Limit to once per minute
            self.context_reset_count += 1
            if self.context_reset_count > 3:  # If we're resetting too often
                self.logger.warning(
                    "Too many context resets in a short time, skipping reset")
                return
        else:
            # Reset counter if enough time has passed
            self.context_reset_count = 0

        self.last_context_reset_time = now

        try:
            # Unload all tensors from GPU
            torch.cuda.empty_cache()
            gc.collect()

            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()

            # Force a new CUDA context by creating and deleting a small tensor
            for i in range(self.gpu_count):
                # Skip if this device was recently reset
                if i == self.last_device_reset and now - self.last_context_reset_time < 300:
                    self.logger.info(
                        f"Skipping reset of device {i} as it was recently reset")
                    continue

                try:
                    # Create and delete a small tensor to reinitialize context
                    with torch.cuda.device(i):
                        temp = torch.ones(1, device=f"cuda:{i}")
                        del temp
                    self.last_device_reset = i
                except Exception as e:
                    self.logger.error(f"Failed to reset device {i}: {e}")

            self.logger.info("CUDA context reset completed")
        except Exception as e:
            self.logger.error(f"Error during CUDA context reset: {e}")

    def force_set_device(self, device_idx: int) -> None:
        """
        Force set the current device to a specific GPU index.

        Args:
            device_idx: Index of the GPU to set as current device.
        """
        if not self.has_gpu or device_idx < 0 or device_idx >= self.gpu_count:
            raise ValueError(f"Invalid device index: {device_idx}")

        self.current_device_idx = device_idx
        self.device = self.devices[device_idx]

        try:
            # Explicitly set current device in torch.cuda
            torch.cuda.set_device(device_idx)
        except Exception as e:
            self.logger.warning(f"Error setting CUDA device: {e}")
            # Attempt to reset CUDA context and try again
            self._reset_cuda_context()
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
            self.memory_stats = {
                "cpu": new_dev_stats()
            }
            return self.memory_stats

        gpus = nvsmi.get_gpus()

        for gpu in gpus:
            self.memory_stats[gpu.id] = self.update_memory_stats(gpu)
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
                    # Try one more time after reset
                    try:
                        return self.update_memory_stats(gpu)
                    except Exception as e2:
                        self.logger.error(
                            f"Still failed after context reset: {e2}")
                self.logger.error(
                    f"Error getting memory stats for device {gpu}: {e}")
        return device_stats

    def estimate_largest_free_block(self, device_idx: int) -> int:
        """
        Estimate the largest contiguous block of free memory.
        Enhanced with better error handling and multiple approaches.
        """
        if not self.has_gpu:
            return 0

        # First approach - use max_memory_allocated as a baseline
        try:
            # Get current memory stats
            free_memory = torch.cuda.get_device_properties(
                device_idx).total_memory - torch.cuda.memory_allocated(device_idx)

            # The largest block may be approximately 70-90% of free memory depending on fragmentation
            # We'll use a conservative estimate and refine with testing if needed
            estimated_largest_block = int(free_memory * 0.7)

            # If there's significant fragmentation, we might need to be even more conservative
            if torch.cuda.memory_allocated(device_idx) > 0.5 * torch.cuda.get_device_properties(device_idx).total_memory:
                # More fragmentation likely when GPU is more than half full
                estimated_largest_block = int(free_memory * 0.6)

            # If binary search testing is needed for more accuracy
            if estimated_largest_block > 1024 * 1024 * 1024:  # Only test if estimated block > 1GB
                # Do binary testing for more accuracy
                return self._binary_search_largest_block(device_idx, free_memory)

            return estimated_largest_block

        except Exception as e:
            self.logger.warning(f"Error estimating largest free block: {e}")
            # Return a conservative estimate
            try:
                free_memory = torch.cuda.get_device_properties(
                    device_idx).total_memory - torch.cuda.memory_allocated(device_idx)
                return int(free_memory * 0.5)  # Very conservative fallback
            except:
                return 0

    def _binary_search_largest_block(self, device_idx: int, free_memory: int) -> int:
        """Use binary search to find the largest block size that can be allocated."""
        # Skip actual testing if we're low on memory
        if free_memory < 1024 * 1024 * 100:  # Less than 100MB
            return int(free_memory * 0.8)  # Conservative estimate

        try:
            with torch.cuda.device(device_idx):
                # Start with 70% of free memory as upper bound
                high = int(free_memory * 0.7)
                low = 1024 * 1024  # 1MB

                # Only do limited iterations to avoid spending too much time
                max_attempts = 8
                attempts = 0

                current_max = 0

                # Binary search with limited iterations
                while low <= high and attempts < max_attempts:
                    attempts += 1
                    mid = (low + high) // 2

                    try:
                        # Try to allocate memory
                        test_tensor = torch.empty(
                            mid, dtype=torch.uint8, device=f"cuda:{device_idx}")
                        # If successful, record and try larger
                        current_max = mid
                        del test_tensor
                        low = mid + 1
                    except torch.cuda.OutOfMemoryError:
                        # If failed, try smaller
                        high = mid - 1

                torch.cuda.empty_cache()
                gc.collect()

                # Return the maximum successful size, or the conservative estimate if all failed
                return current_max if current_max > 0 else int(free_memory * 0.5)

        except Exception as e:
            self.logger.warning(
                f"Error in binary search for largest block: {e}")
            return int(free_memory * 0.5)  # Conservative fallback

    # def get_memory_status_str(self, device_idx: Optional[int] = None) -> str:
    #     """Get a formatted string with memory status."""
    #     if not self.has_gpu:
    #         return "CPU mode (no GPU)"
    #     # If no device specified, show all devices or current device
    #     if device_idx is None:
    #         if self.gpu_count == 1:
    #             return self._format_device_memory_str(0)
    #         else:
    #             return self._format_all_devices_memory_str()

    #     # Show specific device
    #     return self._format_device_memory_str(device_idx)

    # def _format_device_memory_str(self, device_idx: int) -> str:
    #     """Format memory string for a specific device."""
    #     stats = self.update_memory_stats(device_idx)

    #     try:
    #         memory_info = (f"GPU {device_idx} memory - "
    #                        f"Total: {self.format_bytes(stats.mem_total)}, "
    #                        f"Used: {self.format_bytes(stats.mem_used)}, "
    #                        f"Free: {self.format_bytes(stats.mem_free)}, "
    #                        f"Utilization: {stats.mem_util}%")

    #         # Add fragmentation info if available
    #         # if 'largest_block' in stats:
    #         #     memory_info += (f", Largest block: {self.format_bytes(stats['largest_block'])}, "
    #         #                     f"Fragmentation: {stats['fragmentation_percent']}%")

    #         return memory_info
    #     except KeyError:
            return f"GPU {device_idx}: Missing memory stats"

    # def _format_all_devices_memory_str(self) -> str:
    #     """Format memory string for all devices."""
    #     self.update_all_memory_stats()
    #     result = []
    #     for i in range(self.gpu_count):
    #         result.append(self._format_device_memory_str(i))
    #     return "\n".join(result)

    def check_memory_available(self, required_bytes: float) -> bool:
        """
        Check if there is enough memory available on any GPU for the required amount.

        Args:
            required_bytes (float): The amount of memory required in bytes

        Returns:
            bool: True if there is enough memory available, False otherwise
        """
        if not self.has_gpu:
            self.logger.warning("No GPU available for memory check")
            return False

        for (name, stats) in self.update_all_memory_stats().items():
            try:
                # Get total and free memory
                total_mem = stats.mem_total * 1024 * 1024  # Convert MB to bytes
                free_mem = stats.mem_free * 1024 * 1024

                # Apply a safety margin (80% of free memory)
                available_mem = free_mem * 0.8

                self.logger.debug(
                    f"GPU {name}: Total: {total_mem/1e9:.2f}GB, Free: {free_mem/1e9:.2f}GB, Required: {required_bytes/1e9:.2f}GB")

                if available_mem >= required_bytes:
                    return True
            except Exception as e:
                self.logger.error(f"Error checking memory on GPU {name}: {e}")
                continue

        self.logger.warning(
            f"Not enough memory available. Required: {required_bytes/1e9:.2f}GB")
        return False

    def clear_memory(self, device_idx: Optional[int] = None, aggressive: bool = False) -> None:
        """
        Clear unused memory and run garbage collection.

        Args:
            device_idx: Optional specific device to clear, otherwise clear all
            aggressive: If True, perform more aggressive memory clearing
        """
        if not self.has_gpu:
            return

        # Run garbage collection first (affects all devices)
        for i, dev in enumerate(list(gc.garbage)):
            self.logger.warning(f"Garbage {i}: {dev}")
            del dev

        # Clear CUDA cache for specific device or all devices
        if device_idx is not None:
            if aggressive:
                try:
                    # Check if we need to reset the device context completely
                    if device_idx in self.device_oom_count and self.device_oom_count[device_idx] >= 2:
                        self.logger.warning(
                            f"Multiple OOM errors on device {device_idx}, attempting context reset")
                        self._reset_cuda_context()  # Try full context reset
                except Exception as e:
                    self.logger.warning(f"Final cleanup step failed: {str(e)}")
                # Before aggressive clearing, try to identify which modules are using memory

                try:
                    allocated = torch.cuda.memory_allocated(
                        device_idx) / (1024 ** 3)
                    self.logger.info(
                        f"GPU {device_idx} has {allocated:.2f}GB allocated before clearing")

                    # Try to release excessive memory by creating and deleting arrays
                    # This can help reduce fragmentation
                    self.logger.info(
                        f"Performing aggressive memory clearing for GPU {device_idx}")
                    with torch.cuda.device(device_idx):
                        # Attempt to allocate and free memory to trigger defragmentation
                        # Start with small allocations
                        for size_mb in [1, 4, 16, 64, 256]:
                            try:
                                size_bytes = size_mb * 1024 * 1024
                                test_tensor = torch.zeros(
                                    size_bytes // 4, dtype=torch.bfloat16, device=f"cuda:{device_idx}")
                                del test_tensor
                            except Exception as e:
                                self.logger.warning(
                                    f"Failed allocation of {size_mb}MB: {str(e)}")
                                break

                        # Try a larger allocation if smaller ones worked
                        try:
                            free_mem = torch.cuda.get_device_properties(
                                device_idx).total_memory - torch.cuda.memory_allocated(device_idx)
                            if free_mem > 1024 * 1024 * 1024:  # At least 1GB free
                                # Try to allocate 50% of free memory
                                test_size = int(free_mem * 0.5)
                                test_tensor = torch.zeros(
                                    test_size // 4, dtype=torch.bfloat16, device=f"cuda:{device_idx}")
                                del test_tensor
                        except Exception:
                            pass  # Ignore if the large allocation fails

                except Exception as e:
                    self.logger.warning(
                        f"Aggressive memory analysis failed: {e}")

            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
        else:
            # For all devices, attempt aggressive clearing if requested
            if aggressive:
                for i in range(self.gpu_count):
                    self.clear_memory(i, aggressive=True)
            else:
                torch.cuda.empty_cache()  # Clears all devices

            self.update_all_memory_stats()

    # def select_optimal_device(self, force_rescan: bool = False) -> torch.device:
    #     """
    #     Select the GPU with the most available memory.

    #     Args:
    #         force_rescan: If True, force a complete rescan of all devices
    #     """
    #     if not self.has_gpu:
    #         self.logger.info("No GPU available, using CPU")
    #         return torch.device("cpu")

    #     if self.gpu_count == 1:
    #         self.logger.info(
    #             f"Only one GPU available, using device {self.devices[0]}")
    #         return self.devices[0]

    #     # Find device with most free memory
    #     self.update_all_memory_stats()
    #     max_free = -1
    #     optimal_device_idx = 0

    #     for i in range(self.gpu_count):
    #         # Skip blacklisted devices
    #         if i in self.blacklisted_devices:
    #             continue

    #         stats = self.memory_stats.get(i, new_dev_stats())
    #         free_gb = stats.free_gb

    #         # Prioritize largest contiguous block if fragmentation is an issue
    #         if (
    #             isinstance(free_gb, (int, float)) and
    #             isinstance(max_free, (int, float)) and
    #             free_gb > max_free
    #         ):
    #             max_free = free_gb
    #             optimal_device_idx = i

    #     # If we're switching devices, log the change
    #     if optimal_device_idx != self.current_device_idx:
    #         old_device = self.current_device_idx
    #         self.current_device_idx = optimal_device_idx
    #         self.device = self.devices[optimal_device_idx]

    #         # Explicitly set current device in torch.cuda
    #         try:
    #             torch.cuda.set_device(optimal_device_idx)
    #         except Exception as e:
    #             self.logger.warning(f"Error setting CUDA device: {e}")

    #         # Add previous device to fallback list
    #         if old_device not in self.fallback_devices:
    #             self.fallback_devices.append(old_device)
    #     else:
    #         self.logger.info(
    #             f"Keeping GPU {optimal_device_idx} as optimal device with {max_free:.2f}GB free, "
    #             f"{max_largest_block:.2f}GB contiguous block")

    #     return self.device

    # def handle_out_of_memory_error(self, device_idx: Optional[int] = None) -> torch.device:
    #     """
    #     Handle out-of-memory error by selecting a different device if available.

    #     Returns:
    #         A torch device to try (may be the same if no alternatives)
    #     """
    #     if not self.has_gpu:
    #         self.logger.warning(
    #             "Out of memory in CPU mode, cannot switch devices")
    #         return torch.device("cpu")

    #     # If no device specified, use current device
    #     if device_idx is None:
    #         device_idx = self.current_device_idx

    #     # Update OOM counters
    #     self.oom_count += 1
    #     self.device_oom_count[device_idx] = self.device_oom_count.get(
    #         device_idx, 0) + 1
    #     self.last_oom_time = time.time()

    #     # If we've had multiple OOM errors across all devices, reset CUDA context
    #     if self.oom_count >= 3 and (time.time() - self.last_oom_time) < 300:
    #         self.logger.warning(
    #             "Multiple OOM errors detected, resetting CUDA context")
    #         self._reset_cuda_context()

    #     # If this device has had multiple OOMs, consider blacklisting temporarily
    #     if self.device_oom_count[device_idx] >= 3:
    #         if device_idx not in self.blacklisted_devices:
    #             self.logger.warning(
    #                 f"Blacklisting GPU {device_idx} after {self.device_oom_count[device_idx]} OOM errors")
    #             self.blacklisted_devices.append(device_idx)

    #     # Try to recover memory - more aggressive clearing
    #     self.logger.warning(
    #         f"Out of memory on GPU {device_idx}, attempting recovery...")
    #     self.clear_memory(device_idx, aggressive=True)

    #     # If we're in multi-GPU mode and this is the current device, try to switch
    #     if self.gpu_count > 1:
    #         # Find an alternative device (not blacklisted and preferably not in fallback list)
    #         alternative_devices = [i for i in range(self.gpu_count)
    #                                if i != device_idx and i not in self.blacklisted_devices]

    #         if alternative_devices:
    #             # Update memory stats to get latest
    #             self.update_all_memory_stats()

    #             # Find device with most free memory among alternatives
    #             max_free = -1
    #             best_alt_device = alternative_devices[0]

    #             for i in alternative_devices:
    #                 stats = self.memory_stats.get(i, {})
    #                 free_gb = stats.get('free_gb', 0)
    #                 if isinstance(free_gb, (int, float)) and isinstance(max_free, (int, float)) and free_gb > max_free:
    #                     max_free = free_gb
    #                     best_alt_device = i

    #             # Switch to alternative device
    #             old_device = self.current_device_idx
    #             self.current_device_idx = best_alt_device
    #             self.device = self.devices[best_alt_device]

    #             # Explicitly set current device in torch.cuda
    #             try:
    #                 torch.cuda.set_device(best_alt_device)
    #             except Exception as e:
    #                 self.logger.warning(f"Error setting CUDA device: {e}")

    #             self.logger.info(
    #                 f"Switching from GPU {old_device} to alternative GPU {best_alt_device} "
    #                 f"with {max_free:.2f}GB free after OOM error")
    #             return self.device
    #         else:
    #             self.logger.warning(
    #                 "No alternative devices available, staying on current device")

    #     # If we couldn't switch devices, try some extreme memory saving
    #     if device_idx == self.current_device_idx:
    #         try:
    #             # Reset the device context completely as last resort
    #             self.logger.warning("Performing emergency context reset")
    #             self._reset_cuda_context()
    #             self.clear_memory(device_idx, aggressive=True)

    #             # Remove from blacklist temporarily if this was our only option
    #             if device_idx in self.blacklisted_devices and self.gpu_count <= len(self.blacklisted_devices):
    #                 self.logger.warning(
    #                     f"Removing GPU {device_idx} from blacklist as we have no alternatives")
    #                 self.blacklisted_devices.remove(device_idx)
    #         except Exception as e:
    #             self.logger.error(
    #                 f"Failed emergency memory recovery: {str(e)}")

    #     # If we couldn't switch, just return current device
    #     return self.device

    # def is_low_memory(self, threshold_gb: float = 1.0, device_idx: Optional[int] = None) -> bool:
    #     """
    #     Check if available memory is below threshold.

    #     Now also checks for fragmentation - reports low memory if largest contiguous
    #     block is below threshold even if total free memory is higher.
    #     """
    #     if not self.has_gpu:
    #         self.logger.info(f"Running in CPU mode, memory check skipped")
    #         return False

    #     # Convert threshold to bytes
    #     threshold_bytes = int(threshold_gb * 1024 * 1024 * 1024)

    #     # Use current device if none specified
    #     if device_idx is None:
    #         device_idx = self.current_device_idx

    #     self.update_memory_stats(device_idx)
    #     stats = self.memory_stats.get(device_idx, {})
    #     free_bytes = stats.get('free', 0)
    #     largest_block = stats.get('largest_block', 0)

    #     is_low = False

    #     # Check both total free memory and largest contiguous block
    #     if isinstance(free_bytes, int) and isinstance(largest_block, int):
    #         is_low = free_bytes < threshold_bytes or largest_block < threshold_bytes

    #         if is_low:
    #             if largest_block < threshold_bytes:
    #                 self.logger.warning(
    #                     f"Memory fragmentation detected! Largest contiguous block is only "
    #                     f"{self.format_bytes(largest_block)}, threshold is {self.format_bytes(threshold_bytes)}")
    #             else:
    #                 self.logger.warning(
    #                     f"Low memory detected! Only {self.format_bytes(free_bytes)} available, "
    #                     f"threshold is {self.format_bytes(threshold_bytes)}")
    #     else:
    #         self.logger.warning(
    #             f"Could not determine free memory, assuming low memory condition")
    #         is_low = True

    #     return is_low

    # def get_optimal_dimensions(self) -> Tuple[int, int]:
    #     """Get optimal dimensions for image generation based on available memory."""
    #     if not self.has_gpu:
    #         self.logger.info(
    #             "No GPU detected, using default dimensions for CPU (512x512)")
    #         return (512, 512)  # Default lower size for CPU

    #     # Update stats
    #     self.update_memory_stats()
    #     try:
    #         # First try to select optimal device
    #         self.select_optimal_device()

    #         # Use the largest contiguous block size instead of total free memory
    #         largest_block_gb = self.memory_stats[self.current_device_idx].get(
    #             'largest_block_gb', 0)
    #         free_gb = self.memory_stats[self.current_device_idx].get(
    #             'free_gb', 0)

    #         # Use whichever is smaller - this accounts for fragmentation
    #         effective_free_gb = min(largest_block_gb, free_gb)

    #         if not isinstance(effective_free_gb, (int, float)):
    #             self.logger.warning(
    #                 f"Could not determine free memory amount, using conservative dimensions")
    #             effective_free_gb = 0

    #         # Scale dimensions based on available memory
    #         dimensions = (512, 512)  # Default conservative dimensions
    #         if effective_free_gb >= 8.0:
    #             dimensions = (1024, 1024)  # High memory: full resolution
    #             self.logger.info(
    #                 f"High memory available ({effective_free_gb:.2f}GB), using dimensions {dimensions}")
    #         elif effective_free_gb >= 4.0:
    #             dimensions = (768, 768)    # Medium memory
    #             self.logger.info(
    #                 f"Medium memory available ({effective_free_gb:.2f}GB), using dimensions {dimensions}")
    #         elif effective_free_gb >= 2.0:
    #             dimensions = (640, 640)    # Lower memory
    #             self.logger.info(
    #                 f"Lower memory available ({effective_free_gb:.2f}GB), using dimensions {dimensions}")
    #         else:
    #             dimensions = (512, 512)    # Very low memory
    #             self.logger.info(
    #                 f"Low memory available ({effective_free_gb:.2f}GB), using dimensions {dimensions}")

    #         return dimensions

    #     except (KeyError, TypeError) as e:
    #         self.logger.error(f"Error determining optimal dimensions: {e}")
    #         # Default to conservative settings if we can't determine memory
    #         return (512, 512)

    # def get_optimal_inference_steps(self) -> int:
    #     """Get optimal inference steps based on available memory."""
    #     if not self.has_gpu:
    #         self.logger.info(
    #             "No GPU detected, using default inference steps for CPU (20)")
    #         return 20  # Lower default for CPU

    #     # Update stats
    #     self.update_memory_stats()

    #     # Use the largest contiguous block instead of total free memory
    #     largest_block_gb = self.memory_stats[self.current_device_idx].get(
    #         'largest_block_gb', 0)
    #     free_gb = self.memory_stats[self.current_device_idx].get('free_gb', 0)

    #     # Use whichever is smaller
    #     effective_free_gb = min(largest_block_gb, free_gb)

    #     if not isinstance(effective_free_gb, (int, float)):
    #         self.logger.warning(
    #             f"Could not determine free memory amount, using conservative inference steps")
    #         effective_free_gb = 0

    #     # Determine steps based on free memory
    #     steps = 20  # Default conservative steps
    #     if effective_free_gb >= 6.0:
    #         steps = 30  # High quality
    #         self.logger.info(
    #             f"High memory available ({effective_free_gb:.2f}GB), using {steps} inference steps")
    #     elif effective_free_gb >= 3.0:
    #         steps = 25  # Medium quality
    #         self.logger.info(
    #             f"Medium memory available ({effective_free_gb:.2f}GB), using {steps} inference steps")
    #     else:
    #         steps = 20  # Low quality/faster
    #         self.logger.info(
    #             f"Low memory available ({effective_free_gb:.2f}GB), using {steps} inference steps")

    #     return steps

    # def optimize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Optimize generation parameters based on available resources.
    #     Enhanced with more SD3.5-specific optimizations.
    #     """
    #     if not params:
    #         params = {}

    #     original_params = params.copy()

    #     # Always select optimal device before optimizing parameters
    #     self.select_optimal_device()

    #     # Only override if not explicitly specified
    #     if 'width' not in params or 'height' not in params:
    #         optimal_width, optimal_height = self.get_optimal_dimensions()
    #         if 'width' not in params:
    #             params['width'] = optimal_width
    #         if 'height' not in params:
    #             params['height'] = optimal_height

    #     if 'num_inference_steps' not in params:
    #         params['num_inference_steps'] = self.get_optimal_inference_steps()

    #     # If memory is very low, force optimizations
    #     if self.is_low_memory(0.5):  # Less than 500MB free
    #         self.logger.warning(
    #             "WARNING: Very low GPU memory, forcing optimized parameters")
    #         old_width = params.get('width', 512)
    #         old_height = params.get('height', 512)
    #         old_steps = params.get('num_inference_steps', 20)

    #         params['width'] = min(old_width, 512)
    #         params['height'] = min(old_height, 512)
    #         params['num_inference_steps'] = min(old_steps, 20)

    #         if old_width != params['width'] or old_height != params['height'] or old_steps != params['num_inference_steps']:
    #             self.logger.warning(f"Parameters adjusted due to low memory: "
    #                                 f"dimensions {old_width}x{old_height} -> {params['width']}x{params['height']}, "
    #                                 f"steps {old_steps} -> {params['num_inference_steps']}")

    #     # Log parameter changes
    #     if params != original_params:
    #         changes = []
    #         for k, v in params.items():
    #             if k not in original_params:
    #                 changes.append(f"{k}: set to {v}")
    #             elif original_params[k] != v:
    #                 changes.append(f"{k}: {original_params[k]} -> {v}")

    #         if changes:
    #             self.logger.info(
    #                 f"Optimized generation parameters: {', '.join(changes)}")
    #     else:
    #         self.logger.info(
    #             f"Using original generation parameters (no optimization needed)")

    #     return params


# Create a singleton instance
hardware_manager = HardwareManager()
