import os
import torch
import gc
import logging
from typing import Dict, List, Tuple, Optional, Any, Union


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

        if self.has_gpu:
            self.logger.info(f"Detected {self.gpu_count} GPU(s)")
            for i in range(self.gpu_count):
                device = torch.device(f"cuda:{i}")
                self.devices.append(device)
                gpu_name = torch.cuda.get_device_name(i)
                self.logger.info(f"  GPU {i}: {gpu_name}")

            # Select first device as default
            self.device = self.devices[0]
        else:
            self.device = torch.device("cpu")
            self.logger.warning("No GPU detected, running in CPU mode")

        # Initialize memory stats for each device
        self.memory_stats: Dict[int, Dict[str, Union[float, str]]] = {}

        # Apply global PyTorch memory settings
        if self.has_gpu:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            # Log initial state
            self.update_all_memory_stats()
            self.logger.info(
                f"Initialized HardwareManager with primary device: {self.device}")
            self.logger.info(
                f"Initial memory status: {self.get_memory_status_str()}")
        else:
            self.logger.info("No GPU detected, running in CPU mode")

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

            # Store actual byte values for accurate calculations
            device_stats = {
                "total": total_memory,
                "used": allocated_memory,
                "reserved": reserved_memory,
                "free": free_memory,
                "free_reserved": free_reserved,
                "utilization_percent": round((allocated_memory / total_memory) * 100, 1)
            }

            # Store calculated GB values for easier use elsewhere
            device_stats["total_gb"] = total_memory / (1024**3)
            device_stats["used_gb"] = allocated_memory / (1024**3)
            device_stats["free_gb"] = free_memory / (1024**3)
            device_stats["reserved_gb"] = reserved_memory / (1024**3)

            # Store in the memory_stats dict with device index as key
            self.memory_stats[device_idx] = device_stats
            return device_stats

        except Exception as e:
            self.logger.error(
                f"Error getting memory stats for device {device_idx}: {e}")
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
            self.logger.warning(
                f"Garbage could not collect {unreachable_devs} unreachable objects")

        for i, dev in gc.garbage:
            self.logger.warning(f"Garbage {i}: {dev}")

        # Clear CUDA cache for specific device or all devices
        if device_idx is not None:
            with torch.cuda.device(device_idx):
                torch.cuda.empty_cache()
            self.update_memory_stats(device_idx)
            self.logger.info(
                f"Cleared GPU {device_idx} memory. {self._format_device_memory_str(device_idx)}")
        else:
            # Run garbage collection first (affects all devices)
            torch.cuda.empty_cache()  # Clears all devices
            self.update_all_memory_stats()
            self.logger.info(
                f"Cleared all GPU memory. {self.get_memory_status_str()}")

    def select_optimal_device(self) -> torch.device:
        """Select the GPU with the most available memory."""
        if not self.has_gpu:
            self.logger.info("No GPU available, using CPU")
            return torch.device("cpu")

        if self.gpu_count == 1:
            self.logger.info(
                f"Only one GPU available, using device {self.devices[0]}")
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
        self.logger.info(
            f"Selected GPU {optimal_device_idx} as optimal device with {max_free:.2f} GB free memory")
        return self.device

    def is_low_memory(self, threshold_gb: float = 1.0, device_idx: Optional[int] = None) -> bool:
        """Check if available memory is below threshold."""
        if not self.has_gpu:
            self.logger.info(f"Running in CPU mode, memory check skipped")
            return False

        # Convert threshold to bytes
        threshold_bytes = int(threshold_gb * 1024 * 1024 * 1024)

        # Use current device if none specified
        if device_idx is None:
            device_idx = self.current_device_idx

        self.update_memory_stats(device_idx)
        stats = self.memory_stats.get(device_idx, {})
        free_bytes = stats.get('free', 0)

        is_low = False
        if isinstance(free_bytes, int):
            is_low = free_bytes < threshold_bytes
            if is_low:
                self.logger.warning(f"Low memory detected! Only {self.format_bytes(free_bytes)} available, "
                                    f"threshold is {self.format_bytes(threshold_bytes)}")
        else:
            self.logger.warning(
                f"Could not determine free memory, assuming low memory condition")
            is_low = True

        return is_low

    def get_optimal_dimensions(self) -> Tuple[int, int]:
        """Get optimal dimensions for image generation based on available memory."""
        if not self.has_gpu:
            self.logger.info(
                "No GPU detected, using default dimensions for CPU (512x512)")
            return (512, 512)  # Default lower size for CPU

        # Update stats
        self.update_memory_stats()
        try:
            # Use the pre-calculated GB value
            free_gb = self.memory_stats[self.current_device_idx].get(
                'free_gb', 0)

            if not isinstance(free_gb, (int, float)):
                self.logger.warning(
                    f"Could not determine free memory amount, using conservative dimensions")
                free_gb = 0

            # Scale dimensions based on available memory
            dimensions = (512, 512)  # Default conservative dimensions
            if free_gb >= 8.0:
                dimensions = (1024, 1024)  # High memory: full resolution
                self.logger.info(
                    f"High memory available ({free_gb:.2f}GB), using dimensions {dimensions}")
            elif free_gb >= 4.0:
                dimensions = (768, 768)    # Medium memory
                self.logger.info(
                    f"Medium memory available ({free_gb:.2f}GB), using dimensions {dimensions}")
            elif free_gb >= 2.0:
                dimensions = (640, 640)    # Lower memory
                self.logger.info(
                    f"Lower memory available ({free_gb:.2f}GB), using dimensions {dimensions}")
            else:
                dimensions = (512, 512)    # Very low memory
                self.logger.info(
                    f"Low memory available ({free_gb:.2f}GB), using dimensions {dimensions}")

            return dimensions

        except (KeyError, TypeError) as e:
            self.logger.error(f"Error determining optimal dimensions: {e}")
            # Default to conservative settings if we can't determine memory
            return (512, 512)

    def get_optimal_inference_steps(self) -> int:
        """Get optimal inference steps based on available memory."""
        if not self.has_gpu:
            self.logger.info(
                "No GPU detected, using default inference steps for CPU (20)")
            return 20  # Lower default for CPU

        # Update stats
        self.update_memory_stats()

        # Use the pre-calculated GB value
        free_gb = self.memory_stats[self.current_device_idx].get('free_gb', 0)

        if not isinstance(free_gb, (int, float)):
            self.logger.warning(
                f"Could not determine free memory amount, using conservative inference steps")
            free_gb = 0

        # Determine steps based on free memory
        steps = 20  # Default conservative steps
        if free_gb >= 6.0:
            steps = 30  # High quality
            self.logger.info(
                f"High memory available ({free_gb:.2f}GB), using {steps} inference steps")
        elif free_gb >= 3.0:
            steps = 25  # Medium quality
            self.logger.info(
                f"Medium memory available ({free_gb:.2f}GB), using {steps} inference steps")
        else:
            steps = 20  # Low quality/faster
            self.logger.info(
                f"Low memory available ({free_gb:.2f}GB), using {steps} inference steps")

        return steps

    def optimize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize generation parameters based on available resources."""
        if not params:
            params = {}

        original_params = params.copy()

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
            self.logger.warning(
                "WARNING: Very low GPU memory, forcing optimized parameters")
            old_width = params.get('width', 512)
            old_height = params.get('height', 512)
            old_steps = params.get('num_inference_steps', 20)

            params['width'] = min(old_width, 512)
            params['height'] = min(old_height, 512)
            params['num_inference_steps'] = min(old_steps, 20)

            if old_width != params['width'] or old_height != params['height'] or old_steps != params['num_inference_steps']:
                self.logger.warning(f"Parameters adjusted due to low memory: "
                                    f"dimensions {old_width}x{old_height} -> {params['width']}x{params['height']}, "
                                    f"steps {old_steps} -> {params['num_inference_steps']}")

        # Log parameter changes
        if params != original_params:
            changes = []
            for k, v in params.items():
                if k not in original_params:
                    changes.append(f"{k}: set to {v}")
                elif original_params[k] != v:
                    changes.append(f"{k}: {original_params[k]} -> {v}")

            if changes:
                self.logger.info(
                    f"Optimized generation parameters: {', '.join(changes)}")
        else:
            self.logger.info(
                f"Using original generation parameters (no optimization needed)")

        return params


# Create a singleton instance
hardware_manager = HardwareManager()
