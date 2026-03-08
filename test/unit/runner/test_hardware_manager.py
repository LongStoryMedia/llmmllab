"""
Unit tests for runner/utils/hardware_manager.py.

Tests GPU detection, memory management, thermal monitoring, and process management.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
from unittest.mock import Mock
import torch
import time

from runner.utils.hardware_manager import (
    EnhancedHardwareManager,
    MemoryConfig,
    CUDAContextManager,
    GPUProcessManager,
    MemoryManager,
    is_memory_related_error,
    hardware_manager,
)


class TestIsMemoryRelatedError:
    """Tests for is_memory_related_error function."""

    def test_memory_error_keywords(self):
        """Test detection of memory-related error keywords."""
        assert is_memory_related_error("Out of memory") is True
        assert is_memory_related_error("CUDA out of memory") is True
        assert is_memory_related_error("Memory allocation failed") is True
        assert is_memory_related_error("CUDA error: out of memory") is True
        assert is_memory_related_error("Insufficient memory") is True
        assert is_memory_related_error("OOM") is True

    def test_non_memory_error(self):
        """Test non-memory errors are not detected."""
        assert is_memory_related_error("Connection refused") is False
        assert is_memory_related_error("Model not found") is False
        assert is_memory_related_error("Invalid input") is False


class TestMemoryConfig:
    """Tests for MemoryConfig class."""

    def test_memory_config_defaults(self):
        """Test MemoryConfig default values."""
        config = MemoryConfig()

        assert config.safety_margin == 0.8
        assert config.defrag_threshold == 0.9
        assert config.critical_threshold == 0.95
        assert config.context_reset_cooldown == 30

    def test_memory_config_custom_values(self):
        """Test MemoryConfig with custom values."""
        config = MemoryConfig(
            safety_margin=0.9,
            defrag_threshold=0.85,
            critical_threshold=0.9,
            context_reset_cooldown=60,
        )

        assert config.safety_margin == 0.9
        assert config.defrag_threshold == 0.85
        assert config.critical_threshold == 0.9
        assert config.context_reset_cooldown == 60


class TestCUDAContextManager:
    """Tests for CUDAContextManager class."""

    def test_init(self):
        """Test CUDAContextManager initialization."""
        mock_logger = MagicMock()
        manager = CUDAContextManager(mock_logger)

        assert manager.logger == mock_logger
        assert manager.last_reset_time == 0
        assert manager.reset_count == 0
        assert manager.context_initialized == {}

    def test_destroy_context_success(self, mocker):
        """Test successful context destruction."""
        mock_logger = MagicMock()
        manager = CUDAContextManager(mock_logger)

        # Mock ctypes successfully
        mock_cuda = MagicMock()
        mock_cuda.cudaSetDevice = MagicMock()
        mock_cuda.cudaDeviceReset = MagicMock(return_value=0)
        mocker.patch('ctypes.CDLL', return_value=mock_cuda)

        result = manager.destroy_context(0)

        assert result is True
        assert manager.context_initialized[0] is False

    def test_destroy_context_fallback(self, mocker):
        """Test context destruction with fallback."""
        mock_logger = MagicMock()
        manager = CUDAContextManager(mock_logger)

        # Mock ctypes failing
        mocker.patch('ctypes.CDLL', side_effect=OSError("Not found"))
        mock_manager = MagicMock()
        mocker.patch.object(manager, '_pytorch_aggressive_reset', mock_manager)

        result = manager.destroy_context(0)

        assert result is True

    def test_reset_context_with_cooldown(self, mocker):
        """Test context reset with cooldown protection."""
        mock_logger = MagicMock()
        manager = CUDAContextManager(mock_logger)
        manager.last_reset_time = time.time()

        # Mock the reset methods
        mock_manager = MagicMock()
        mocker.patch.object(manager, '_try_pynvml_reset', return_value=False)
        mocker.patch.object(manager, '_pytorch_aggressive_reset', mock_manager)

        # First call should succeed
        result1 = manager.reset_context(0, cooldown=60)
        assert result1 is True

        # Second call should be skipped due to cooldown
        result2 = manager.reset_context(0, cooldown=60)
        assert result2 is True


class TestGPUProcessManager:
    """Tests for GPUProcessManager class."""

    def test_get_processes_success(self, mocker):
        """Test getting GPU processes successfully."""
        mock_logger = MagicMock()
        manager = GPUProcessManager(mock_logger)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1234,python,1024\n5678,torch,2048\n"
        mock_result.stderr = ""
        mocker.patch('subprocess.run', return_value=mock_result)

        processes = manager.get_processes(0)

        assert len(processes) == 2
        assert processes[0]["pid"] == 1234
        assert processes[0]["memory_mb"] == 1024

    def test_get_processes_failure(self, mocker):
        """Test getting GPU processes with failure."""
        mock_logger = MagicMock()
        manager = GPUProcessManager(mock_logger)

        mock_result = MagicMock()
        mock_result.returncode = 1
        mocker.patch('subprocess.run', return_value=mock_result)

        processes = manager.get_processes(0)

        assert processes == []

    def test_kill_process_success(self, mocker):
        """Test killing a process successfully."""
        mock_logger = MagicMock()
        manager = GPUProcessManager(mock_logger)

        mock_process = MagicMock()
        mock_process.terminate = MagicMock()
        mocker.patch('psutil.Process', return_value=mock_process)
        mocker.patch('psutil.pid_exists', return_value=True)

        result = manager.kill_process(1234)

        assert result is True

    def test_kill_process_not_exists(self, mocker):
        """Test killing a non-existent process."""
        mock_logger = MagicMock()
        manager = GPUProcessManager(mock_logger)

        mocker.patch('psutil.pid_exists', return_value=False)

        result = manager.kill_process(1234)

        assert result is False


class TestMemoryManager:
    """Tests for MemoryManager class."""

    def test_clear_memory(self, mocker):
        """Test clearing memory."""
        mock_logger = MagicMock()
        config = MemoryConfig()
        manager = MemoryManager(mock_logger, config)

        mock_destroy = mocker.patch.object(
            manager.context_manager, 'destroy_context', return_value=True
        )
        mock_kill = mocker.patch.object(
            manager.process_manager, 'kill_gpu_processes', return_value=0
        )

        manager.clear_memory(0)

        mock_destroy.assert_called_once()
        mock_kill.assert_called_once()


class TestEnhancedHardwareManager:
    """Tests for EnhancedHardwareManager class."""

    def test_init_no_gpu(self, mocker):
        """Test initialization without GPU."""
        mocker.patch('torch.cuda.is_available', return_value=False)

        manager = EnhancedHardwareManager()

        assert manager.has_gpu is False
        assert manager.gpu_count == 0
        assert manager.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    def test_init_with_gpu(self, mocker):
        """Test initialization with GPU."""
        mocker.patch('torch.cuda.is_available', return_value=True)
        mocker.patch('torch.cuda.device_count', return_value=1)
        mocker.patch('torch.cuda.get_device_name', return_value="Test GPU")
        mocker.patch('torch.cuda.empty_cache')
        mocker.patch('torch.cuda.synchronize')

        manager = EnhancedHardwareManager()

        assert manager.has_gpu is True
        assert manager.gpu_count == 1

    def test_check_memory_available_cpu_mode(self):
        """Test memory check in CPU mode."""
        manager = EnhancedHardwareManager()
        manager.has_gpu = False

        result = manager.check_memory_available(1024)

        assert result is False

    def test_check_gpu_thermals_no_gpu(self):
        """Test thermal check without GPU."""
        manager = EnhancedHardwareManager()
        manager.has_gpu = False

        temps = manager.check_gpu_thermals()

        assert temps == {}

    def test_get_device_mappings_no_gpu(self):
        """Test device mappings in CPU mode."""
        manager = EnhancedHardwareManager()
        manager.has_gpu = False

        mappings = manager.get_device_mappings()

        assert "cpu" in mappings
        assert mappings["cpu"]["name"] == "CPU"

    def test_format_bytes(self):
        """Test byte formatting."""
        assert EnhancedHardwareManager.format_bytes(1024) == "1.00 KB"
        assert EnhancedHardwareManager.format_bytes(1024 * 1024) == "1.00 MB"
        assert EnhancedHardwareManager.format_bytes(1024 * 1024 * 1024) == "1.00 GB"