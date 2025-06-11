import os
import torch
import gc
from typing import Dict, List, Tuple, Optional, Any, Union


class HardwareManager:
    """Service for managing hardware resources, particularly GPU memory."""

    def __init__(self):
        """Initialize the hardware manager and discover available GPU resources."""
        self.has_gpu = torch.cuda.is_available()

        # Discover all available GPUs
        self.gpu_count = torch.cuda.device_count() if self.has_gpu else 0
        self.devices: List[torch.device] = []
        self.current_device_idx = 0

        if self.has_gpu:
            print(f"Detected {self.gpu_count} GPU(s)")
            for i in range(self.gpu_count):
                device = torch.device(f"cuda:{i}")
                self.devices.append(device)
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")

            # Select first device as default
            self.device = self.devices[0]
        else:
            self.device = torch.device("cpu")

        # Initialize memory stats for each device
        self.memory_stats: Dict[int, Dict[str, Union[float, str]]] = {}

        # Apply global PyTorch memory settings
        if self.has_gpu:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            # Log initial state
            self.update_all_memory_stats()
            print(
                f"Initialized HardwareManager with primary device: {self.device}")
            print(f"Initial memory status: {self.get_memory_status_str()}")
        else:
            print("No GPU detected, running in CPU mode")

    def update_all_memory_stats(self) -> Dict[int, Dict[str, Union[float, str]]]:
        """Update memory statistics for all devices."""
        if not self.has_gpu:
            self.memory_stats = {-1: {"mode": "cpu"}}
            return self.memory_stats

        for i in range(self.gpu_count):
            self.update_memory_stats(i)
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

    def update_memory_stats(self, device_idx: Optional[int] = None) -> Dict[str, Union[float, str, int]]:
        """Update and return memory statistics for a specific device."""
        if not self.has_gpu:
            self.memory_stats = {-1: {"mode": "cpu"}}
            return {"mode": "cpu"}

        # If no device specified, use the current one
        if device_idx is None:
            device_idx = self.current_device_idx

        # Validate device index
        if device_idx < 0 or device_idx >= self.gpu_count:
            return {"error": f"Invalid device index: {device_idx}"}

        # Get memory information
        try:
            # Get raw byte values from CUDA
            total_memory = torch.cuda.get_device_properties(
                device_idx).total_memory
            allocated_memory = torch.cuda.memory_allocated(device_idx)
            reserved_memory = torch.cuda.memory_reserved(device_idx)
            free_memory = total_memory - allocated_memory
            free_reserved = reserved_memory - allocated_memory

            # class _CudaDeviceProperties:
            # name: str
            # major: _int
            # minor: _int
            # multi_processor_count: _int
            # total_memory: _int
            # is_integrated: _int
            # is_multi_gpu_board: _int
            # max_threads_per_multi_processor: _int
            # gcnArchName: str
            # warp_size: _int
            # uuid: str
            # L2_cache_size: _int

            # Store actual byte values for accurate calculations
            device_stats = {
                "total": total_memory,
                "used": allocated_memory,
                "reserved": reserved_memory,
                "free": free_memory,
                "free_reserved": free_reserved,
                "utilization_percent": round((allocated_memory / total_memory) * 100, 1)
            }

            # Store in the memory_stats dict with device index as key
            self.memory_stats[device_idx] = device_stats
            return device_stats

        except Exception as e:
            print(f"Error getting memory stats for device {device_idx}: {e}")
            error_stats: Dict[str, Union[float, str]] = {"error": str(e)}
            self.memory_stats[device_idx] = error_stats
            return error_stats

    def get_memory_status_str(self, device_idx: Optional[int] = None) -> str:
        """Get a formatted string with memory status."""
        if not self.has_gpu:
            return "CPU mode (no GPU)"
        # If no device specified, show all devices or current device
        if device_idx is None:
            if self.gpu_count == 1:
                return self._format_device_memory_str(0)
            else:
                return self._format_all_devices_memory_str()

        # Show specific device
        return self._format_device_memory_str(device_idx)

    def _format_device_memory_str(self, device_idx: int) -> str:
        """Format memory string for a specific device."""
        stats = self.update_memory_stats(device_idx)
        if "error" in stats:
            return f"GPU {device_idx}: Error - {stats['error']}"

        try:
            return (f"GPU {device_idx} memory - "
                    f"Total: {self.format_bytes(stats['total'])}, "
                    f"Used: {self.format_bytes(stats['used'])}, "
                    f"Free: {self.format_bytes(stats['free'])}, "
                    f"Utilization: {stats['utilization_percent']}%")
        except KeyError:
            return f"GPU {device_idx}: Missing memory stats"

    def _format_all_devices_memory_str(self) -> str:
        """Format memory string for all devices."""
        self.update_all_memory_stats()
        result = []
        for i in range(self.gpu_count):
            result.append(self._format_device_memory_str(i))
        return "\n".join(result)

    def clear_memory(self, device_idx: Optional[int] = None) -> None:
        """Clear unused memory and run garbage collection."""
        if not self.has_gpu:
            return

        # Run garbage collection first (affects all devices)
        unreachable_devs = gc.collect()
        if unreachable_devs > 0:
            print(
                f"Garbage could not collect {unreachable_devs} unreachable objects")

        for i, dev in gc.garbage:
            print(f"Garbage {i}: {dev}")

        # Clear CUDA cache for specific device or all devices
        if device_idx is not None:
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
            self.update_memory_stats(device_idx)
            print(
                f"Cleared GPU {device_idx} memory. {self._format_device_memory_str(device_idx)}")
        else:
            # Run garbage collection first (affects all devices)
            torch.cuda.empty_cache()  # Clears all devices
            self.update_all_memory_stats()
            print(f"Cleared all GPU memory. {self.get_memory_status_str()}")

    def select_optimal_device(self) -> torch.device:
        """Select the GPU with the most available memory."""
        if not self.has_gpu:
            return torch.device("cpu")

        if self.gpu_count == 1:
            return self.devices[0]

        # Find device with most free memory
        self.update_all_memory_stats()
        max_free = -1
        optimal_device_idx = 0

        for i in range(self.gpu_count):
            stats = self.memory_stats.get(i, {})
            free_gb = stats.get('free_gb', 0)
            if isinstance(free_gb, (int, float)) and free_gb > max_free:
                max_free = free_gb
                optimal_device_idx = i

        self.current_device_idx = optimal_device_idx
        self.device = self.devices[optimal_device_idx]
        for i in range(self.gpu_count):
            print(
                f"Selected GPU {optimal_device_idx} as optimal device with {max_free:.2f} GB free memory")
        return self.device

    def is_low_memory(self, threshold_gb: float = 1.0, device_idx: Optional[int] = None) -> bool:
        """Check if available memory is below threshold."""
        if not self.has_gpu:
            return False

        # Convert threshold to bytes
        threshold_bytes = int(threshold_gb * 1024 * 1024 * 1024)

        # Use current device if none specified
        if device_idx is None:
            device_idx = self.current_device_idx

        self.update_memory_stats(device_idx)
        stats = self.memory_stats.get(device_idx, {})
        free_bytes = stats.get('free', 0)

        if isinstance(free_bytes, int):
            return free_bytes < threshold_bytes
        return True  # Assume low memory if free_bytes is not available or not an int

    def get_optimal_dimensions(self) -> Tuple[int, int]:
        """Get optimal dimensions for image generation based on available memory."""
        if not self.has_gpu:
            return (512, 512)  # Default lower size for CPU

        # Update stats
        self.update_memory_stats()
        try:
            free_bytes = self.memory_stats[self.current_device_idx]['free']

            # Convert to GB for human-readable thresholds
            if isinstance(free_bytes, (int, float)):
                free_gb = free_bytes / (1024**3)
            else:
                # Default to conservative value if free_bytes is not numeric
                free_gb = 0

            # Scale dimensions based on available memory
            if free_gb >= 8.0:
                return (1024, 1024)  # High memory: full resolution
            elif free_gb >= 4.0:
                return (768, 768)    # Medium memory
            elif free_gb >= 2.0:
                return (640, 640)    # Lower memory
            else:
                return (512, 512)    # Very low memory
        except (KeyError, TypeError):
            # Default to conservative settings if we can't determine memory
            return (512, 512)

    def get_optimal_inference_steps(self) -> int:
        """Get optimal inference steps based on available memory."""
        if not self.has_gpu:
            return 20  # Lower default for CPU

        # Update stats
        self.update_memory_stats()
        free_bytes = self.memory_stats[self.current_device_idx]['free']

        # Convert to GB
        if isinstance(free_bytes, (int, float)):
            free_gb = free_bytes / (1024**3)
        else:
            # Default to conservative value if free_bytes is not numeric
            free_gb = 0
        free_bytes = self.memory_stats[self.current_device_idx]['free']

        # Convert to GB for human-readable thresholds
        if isinstance(free_bytes, (int, float)):
            free_gb = free_bytes / (1024**3)
        else:
            # Default to conservative value if free_bytes is not numeric
            free_gb = 0

        # Determine steps based on free memory
        if free_gb >= 6.0:
            return 30  # High quality
        elif free_gb >= 3.0:
            return 25  # Medium quality
        else:
            return 20  # Low quality/faster

    def optimize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize generation parameters based on available resources."""
        if not params:
            params = {}

        # Only override if not explicitly specified
        if 'width' not in params or 'height' not in params:
            optimal_width, optimal_height = self.get_optimal_dimensions()
            if 'width' not in params:
                params['width'] = optimal_width
            if 'height' not in params:
                params['height'] = optimal_height

        if 'num_inference_steps' not in params:
            params['num_inference_steps'] = self.get_optimal_inference_steps()

        # If memory is very low, force optimizations
        if self.is_low_memory(0.5):  # Less than 500MB free
            print("WARNING: Very low GPU memory, forcing optimized parameters")
            params['width'] = min(params.get('width', 512), 512)
            params['height'] = min(params.get('height', 512), 512)
            params['num_inference_steps'] = min(
                params.get('num_inference_steps', 20), 20)

        return params


# Create a singleton instance
hardware_manager = HardwareManager()
