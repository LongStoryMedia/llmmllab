"""
Intelligent OOM Recovery System for LLM Pipeline Initialization.

Uses machine learning to predict optimal parameters and implements a structured
retry strategy for handling out-of-memory errors during model initialization.

Features strong typing, dynamic multi-GPU support, sklearn requirement, and
model profile integration.
"""

import os
import json
import hashlib
import numpy as np
from typing import Optional, Tuple, List, TypedDict, Literal, Union
from dataclasses import dataclass, asdict
from pathlib import Path

# sklearn is required - no fallbacks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils.logging import llmmllogger
from models.model_profile import ModelProfile
from models.model_parameters import ModelParameters
from models.gpu_config import GPUConfig


class GPUInfo(TypedDict):
    """Information about a single GPU device."""

    key: str
    id: int
    name: str
    available_memory: float
    total_memory: float
    used_memory: float
    utilization_pct: float


class SystemGPUStats(TypedDict):
    """System-wide GPU statistics."""

    total_gpus: int
    total_memory: float
    available_memory: float
    primary_gpu_id: int
    gpus: List[GPUInfo]


class ModelConfigurationData(TypedDict):
    """Configuration parameters for model initialization."""

    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    model_size_mb: float
    available_gpu_memory_mb: float
    total_gpu_memory_mb: float
    success: bool
    gpu_memory_used_mb: Optional[float]
    initialization_time_ms: Optional[float]


class OOMRecoveryAttemptData(TypedDict):
    """Single OOM recovery attempt configuration."""

    attempt: int
    strategy: Literal["clear_memory", "reduce_batch", "move_to_cpu", "reduce_context"]
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    success: bool
    error_message: str


class PredictionFeatures(TypedDict):
    """Features used for ML prediction."""

    model_size_mb: float
    target_n_ctx: int
    primary_gpu_memory_mb: float
    total_gpu_memory_mb: float
    gpu_count: int
    requested_gpu_layers: int
    memory_pressure_ratio: float
    system_memory_pressure: float
    primary_gpu_fraction: float
    log_model_size: float


class OptimalParameters(TypedDict):
    """Optimal parameters predicted by ML models."""

    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int


class RecoveryStrategy(TypedDict):
    """Recovery strategy result."""

    parameters: OptimalParameters
    strategy_name: Literal[
        "clear_memory", "reduce_batch", "move_to_cpu", "reduce_context"
    ]


class MLModelPerformance(TypedDict):
    """ML model performance metrics."""

    n_ctx_mse: float
    n_batch_mse: float
    n_ubatch_mse: float
    n_gpu_layers_mse: float
    total_configurations: int
    models_trained: bool


class IntelligentOOMRecovery:
    """
    Intelligent OOM recovery system using machine learning to predict optimal parameters.

    Features:
    - Dynamic multi-GPU support for any number of GPUs
    - Strong typing with TypedDict structures
    - sklearn required (Ridge regression models)
    - Model profile integration for configuration-driven algorithms
    - Structured retry strategy: clear memory -> reduce batch -> move to CPU -> reduce context
    - Learning from successful configurations to improve future predictions
    - Persistent storage of training data for cross-session learning
    """

    def __init__(self, data_dir: str = "/tmp/oom_recovery_data") -> None:
        self.logger = llmmllogger.bind(component="IntelligentOOMRecovery")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ML models for predicting optimal parameters (sklearn Ridge regression required)
        self.models: dict[str, Optional[Ridge]] = {
            "n_ctx": None,
            "n_batch": None,
            "n_ubatch": None,
            "n_gpu_layers": None,
        }

        self.scalers: dict[str, StandardScaler] = {
            "features": StandardScaler(),
            "n_ctx": StandardScaler(),
            "n_batch": StandardScaler(),
            "n_ubatch": StandardScaler(),
            "n_gpu_layers": StandardScaler(),
        }

        # Training data storage with strong typing
        self.configurations: List[ModelConfigurationData] = []
        self.recovery_attempts: List[OOMRecoveryAttemptData] = []

        # Load existing data and train models
        self._load_training_data()
        self._train_models()

    def get_model_size_mb(self, model_path: str) -> float:
        """Get model file size in MB with intelligent fallback estimation."""
        if not os.path.exists(model_path):
            # File doesn't exist, use intelligent estimation based on model name patterns
            return self._estimate_model_size_from_name(model_path)
            
        try:
            size_bytes = os.path.getsize(model_path)
            return size_bytes / (1024 * 1024)
        except Exception as e:
            self.logger.warning(f"Error getting model size for {model_path}: {e}")
            return self._estimate_model_size_from_name(model_path)
            
    def _estimate_model_size_from_name(self, model_path: str) -> float:
        """Estimate model size based on filename patterns."""
        model_name = model_path.lower()
        
        # Pattern matching for common model sizes
        if "3b" in model_name or "3.2b" in model_name:
            return 2000.0  # ~2GB for 3B models
        elif "7b" in model_name:
            return 4000.0  # ~4GB for 7B models
        elif "13b" in model_name:
            return 8000.0  # ~8GB for 13B models
        elif "20b" in model_name:
            return 12000.0  # ~12GB for 20B models
        elif "30b" in model_name or "33b" in model_name:
            return 20000.0  # ~20GB for 30B models
        elif "70b" in model_name:
            return 40000.0  # ~40GB for 70B models
        elif "qwen" in model_name:
            return 20000.0  # Default for Qwen models (usually 30B range)
        else:
            return 8000.0  # Conservative default estimate

    def get_system_gpu_stats(self, hardware_manager) -> SystemGPUStats:
        """
        Get comprehensive GPU statistics for dynamic multi-GPU systems.

        Handles any number of GPUs and provides detailed system-wide statistics.
        """
        try:
            memory_stats = hardware_manager.update_all_memory_stats()
            if not memory_stats:
                # Return minimal valid structure for systems without GPUs
                return SystemGPUStats(
                    total_gpus=0,
                    total_memory=0.0,
                    available_memory=0.0,
                    primary_gpu_id=-1,
                    gpus=[],
                )

            # Collect all GPU information
            gpus: List[GPUInfo] = []

            for key, stats in memory_stats.items():
                # Check if this is a GPU entry (DevStats with GPU-like attributes)
                if (
                    hasattr(stats, "mem_total")
                    and hasattr(stats, "mem_free")
                    and hasattr(stats, "name")
                    and "nvidia" in stats.name.lower()
                ):
                    available_memory = max(getattr(stats, "mem_free", 0), 0)
                    total_memory = max(getattr(stats, "mem_total", 0), 0)
                    used_memory = max(getattr(stats, "mem_used", 0), 0)

                    gpu_info: GPUInfo = {
                        "key": str(key),
                        "id": int(getattr(stats, "id", key)),
                        "name": getattr(stats, "name", f"GPU {key}"),
                        "available_memory": available_memory,
                        "total_memory": total_memory,
                        "used_memory": used_memory,
                        "utilization_pct": (
                            (used_memory / total_memory) * 100
                            if total_memory > 0
                            else 0.0
                        ),
                    }
                    gpus.append(gpu_info)

            if not gpus:
                # No GPUs found
                return SystemGPUStats(
                    total_gpus=0,
                    total_memory=0.0,
                    available_memory=0.0,
                    primary_gpu_id=-1,
                    gpus=[],
                )

            # Sort GPUs by ID for consistent ordering
            gpus.sort(key=lambda x: x["id"])

            # Calculate system totals
            total_memory = sum(gpu["total_memory"] for gpu in gpus)
            total_available_memory = sum(gpu["available_memory"] for gpu in gpus)

            # Select primary GPU using intelligent strategy
            primary_gpu_id = self._select_primary_gpu(gpus)

            # Get primary GPU available memory
            primary_gpu_memory = 0.0
            for gpu in gpus:
                if gpu["id"] == primary_gpu_id:
                    primary_gpu_memory = gpu["available_memory"]
                    break

            system_stats: SystemGPUStats = {
                "total_gpus": len(gpus),
                "total_memory": total_memory,
                "available_memory": primary_gpu_memory,  # Primary GPU memory for compatibility
                "primary_gpu_id": primary_gpu_id,
                "gpus": gpus,
            }

            # Log comprehensive GPU information
            gpu_list = ", ".join(
                [
                    f"GPU{gpu['id']}({gpu['available_memory']:.0f}/{gpu['total_memory']:.0f}MB)"
                    for gpu in gpus
                ]
            )

            self.logger.info(
                f"Multi-GPU System: {len(gpus)} GPUs, Total: {total_memory:.0f}MB, "
                f"Primary GPU{primary_gpu_id}: {primary_gpu_memory:.0f}MB, GPUs: [{gpu_list}]"
            )

            return system_stats

        except Exception as e:
            self.logger.warning(f"Error getting GPU system stats: {e}")
            return SystemGPUStats(
                total_gpus=0,
                total_memory=0.0,
                available_memory=0.0,
                primary_gpu_id=-1,
                gpus=[],
            )

    def _select_primary_gpu(self, gpus: List[GPUInfo]) -> int:
        """
        Intelligently select primary GPU from available GPUs.

        Selection priority:
        1. GPU 0 if it has sufficient memory (≥8GB)
        2. GPU with most available memory if GPU 0 is insufficient
        3. GPU 0 for consistency even if not optimal
        4. First GPU if GPU 0 doesn't exist
        """
        if not gpus:
            return -1

        # Find GPU 0
        gpu_0: Optional[GPUInfo] = None
        for gpu in gpus:
            if gpu["id"] == 0:
                gpu_0 = gpu
                break

        # Single GPU case
        if len(gpus) == 1:
            return gpus[0]["id"]

        # Multi-GPU selection logic
        if gpu_0 is not None:
            if gpu_0["available_memory"] < 8000:  # GPU 0 has insufficient memory
                # Select GPU with most available memory
                best_gpu = max(gpus, key=lambda x: x["available_memory"])
                self.logger.info(
                    f"GPU 0 insufficient ({gpu_0['available_memory']:.0f}MB < 8GB), "
                    f"selecting GPU {best_gpu['id']} with {best_gpu['available_memory']:.0f}MB"
                )
                return best_gpu["id"]
            else:
                # GPU 0 has sufficient memory, check if another GPU is significantly better
                best_gpu = max(gpus, key=lambda x: x["available_memory"])

                if (
                    best_gpu["id"] != gpu_0["id"]
                    and best_gpu["available_memory"] > gpu_0["available_memory"] * 2.0
                ):
                    self.logger.info(
                        f"Switching from GPU 0 to GPU {best_gpu['id']} due to significantly more memory "
                        f"({best_gpu['available_memory']:.0f} vs {gpu_0['available_memory']:.0f}MB)"
                    )
                    return best_gpu["id"]
                else:
                    self.logger.info(f"Using GPU 0 for consistency")
                    return gpu_0["id"]
        else:
            # No GPU 0, select GPU with most memory
            best_gpu = max(gpus, key=lambda x: x["available_memory"])
            self.logger.info(
                f"No GPU 0 found, selecting GPU {best_gpu['id']} with most memory"
            )
            return best_gpu["id"]

    def create_configuration_from_model_profile(
        self, model_profile: ModelProfile, gpu_stats: SystemGPUStats
    ) -> OptimalParameters:
        """
        Create initial configuration from ModelProfile, integrating model profile parameters
        with system capabilities for optimal initialization.
        """
        params = model_profile.parameters
        gpu_config = model_profile.gpu_config

        # Extract base parameters from model profile with proper defaults
        base_n_ctx = params.num_ctx or 32768
        base_batch_size = params.batch_size or 512
        base_n_ubatch = base_batch_size  # Default n_ubatch = n_batch

        # GPU layers configuration
        if gpu_config and gpu_config.gpu_layers is not None:
            if gpu_config.gpu_layers == -1:
                # Auto-allocation based on system capabilities
                base_n_gpu_layers = self._estimate_gpu_layers_from_system(gpu_stats)
            else:
                # Explicit configuration from profile
                base_n_gpu_layers = max(0, gpu_config.gpu_layers)
        else:
            # Default: estimate based on available memory
            base_n_gpu_layers = self._estimate_gpu_layers_from_system(gpu_stats)

        # Apply system constraints and optimizations
        optimized_config: OptimalParameters = {
            "n_ctx": min(base_n_ctx, self._max_context_for_memory(gpu_stats)),
            "n_batch": min(base_batch_size, self._max_batch_for_memory(gpu_stats)),
            "n_ubatch": min(base_n_ubatch, base_batch_size),
            "n_gpu_layers": min(
                base_n_gpu_layers, self._max_gpu_layers_for_memory(gpu_stats)
            ),
        }

        self.logger.info(
            f"Profile-driven config: n_ctx={optimized_config['n_ctx']}, "
            f"n_batch={optimized_config['n_batch']}, n_ubatch={optimized_config['n_ubatch']}, "
            f"n_gpu_layers={optimized_config['n_gpu_layers']} (from {model_profile.name})"
        )

        return optimized_config

    def _estimate_gpu_layers_from_system(self, gpu_stats: SystemGPUStats) -> int:
        """Estimate optimal GPU layers based on system capabilities."""
        if gpu_stats["total_gpus"] == 0:
            return 0  # CPU-only

        # Conservative estimate: use layers proportional to available memory
        # Assume ~100MB per layer as rough estimate
        available_memory = gpu_stats["available_memory"]
        estimated_layers = int(available_memory / 100)

        # Reasonable bounds
        return max(0, min(estimated_layers, 128))  # Most models have <128 layers

    def _max_context_for_memory(self, gpu_stats: SystemGPUStats) -> int:
        """Calculate maximum context size for available GPU memory."""
        if gpu_stats["total_gpus"] == 0:
            return 8192  # Conservative for CPU

        # Rough estimate: context uses ~4 bytes per token in KV cache
        available_memory_bytes = gpu_stats["available_memory"] * 1024 * 1024
        # Reserve 50% for model weights, use 50% for context
        context_memory = available_memory_bytes * 0.5
        max_tokens = int(context_memory / 4)

        # Reasonable bounds
        return max(1024, min(max_tokens, 131072))  # 1K to 128K tokens

    def _max_batch_for_memory(self, gpu_stats: SystemGPUStats) -> int:
        """Calculate maximum batch size for available GPU memory."""
        if gpu_stats["total_gpus"] == 0:
            return 64  # Conservative for CPU

        # Conservative batch sizing based on available memory
        available_gb = gpu_stats["available_memory"] / 1024
        if available_gb > 20:
            return 1024
        elif available_gb > 16:
            return 512
        elif available_gb > 12:
            return 256
        elif available_gb > 8:
            return 128
        else:
            return 64

    def _max_gpu_layers_for_memory(self, gpu_stats: SystemGPUStats) -> int:
        """Calculate maximum GPU layers for available memory."""
        return self._estimate_gpu_layers_from_system(gpu_stats)

    def _extract_features(
        self,
        model_size_mb: float,
        gpu_stats: SystemGPUStats,
        target_n_ctx: int,
        requested_gpu_layers: int,
    ) -> PredictionFeatures:
        """
        Extract features for ML prediction with comprehensive multi-GPU awareness.

        Returns structured features optimized for Ridge regression models.
        """
        primary_gpu_memory = gpu_stats["available_memory"]
        total_gpu_memory = gpu_stats["total_memory"]
        gpu_count = gpu_stats["total_gpus"]

        # Calculate derived features
        memory_pressure_ratio = (
            model_size_mb / primary_gpu_memory
            if primary_gpu_memory > 0
            else float("inf")
        )

        system_memory_pressure = (
            model_size_mb / total_gpu_memory if total_gpu_memory > 0 else float("inf")
        )

        primary_gpu_fraction = (
            primary_gpu_memory / total_gpu_memory if total_gpu_memory > 0 else 0.0
        )

        features: PredictionFeatures = {
            "model_size_mb": model_size_mb,
            "target_n_ctx": target_n_ctx,
            "primary_gpu_memory_mb": primary_gpu_memory,
            "total_gpu_memory_mb": total_gpu_memory,
            "gpu_count": gpu_count,
            "requested_gpu_layers": requested_gpu_layers,
            "memory_pressure_ratio": min(
                memory_pressure_ratio, 10.0
            ),  # Cap for numerical stability
            "system_memory_pressure": min(system_memory_pressure, 10.0),
            "primary_gpu_fraction": primary_gpu_fraction,
            "log_model_size": np.log(model_size_mb + 1),
        }

        return features

    def _features_to_array(self, features: PredictionFeatures) -> np.ndarray:
        """Convert feature dictionary to numpy array for sklearn."""
        return np.array(
            [
                features["model_size_mb"],
                features["target_n_ctx"],
                features["primary_gpu_memory_mb"],
                features["total_gpu_memory_mb"],
                features["gpu_count"],
                features["requested_gpu_layers"],
                features["memory_pressure_ratio"],
                features["system_memory_pressure"],
                features["primary_gpu_fraction"],
                features["log_model_size"],
            ]
        )

    def predict_optimal_parameters_from_profile(
        self,
        model_profile: ModelProfile,
        model_path: str,
        hardware_manager,
    ) -> OptimalParameters:
        """
        Predict optimal parameters using model profile configuration and ML models.

        This is the primary method that integrates model profile configuration
        with system capabilities and ML-based optimization.
        """
        # Get system GPU statistics
        gpu_stats = self.get_system_gpu_stats(hardware_manager)

        # Start with model profile configuration
        base_config = self.create_configuration_from_model_profile(
            model_profile, gpu_stats
        )

        # Get model size for ML features
        model_size_mb = self.get_model_size_mb(model_path)

        # Extract features for ML prediction
        features = self._extract_features(
            model_size_mb,
            gpu_stats,
            base_config["n_ctx"],
            base_config["n_gpu_layers"],
        )

        # Apply ML optimization if models are trained
        if all(model is not None for model in self.models.values()):
            optimized_config = self._apply_ml_optimization(features, base_config)
            self.logger.info(f"ML-optimized parameters: {optimized_config}")
        else:
            # Use profile-based configuration with system constraints
            optimized_config = base_config
            self.logger.info(f"Profile-driven parameters: {optimized_config}")

        return optimized_config

    def _apply_ml_optimization(
        self, features: PredictionFeatures, base_config: OptimalParameters
    ) -> OptimalParameters:
        """Apply ML model predictions to optimize base configuration."""
        # Convert features to array for sklearn
        features_array = self._features_to_array(features)
        features_scaled = self.scalers["features"].transform([features_array])

        # Get ML predictions for each parameter
        ml_predictions: dict[str, int] = {}
        for param_name, model in self.models.items():
            if model is not None:
                pred_scaled = model.predict(features_scaled)[0]
                pred_value = self.scalers[param_name].inverse_transform(
                    [[pred_scaled]]
                )[0][0]
                ml_predictions[param_name] = max(int(pred_value), 1)
            else:
                ml_predictions[param_name] = base_config[param_name]  # type: ignore

        # Combine ML predictions with profile constraints
        optimized: OptimalParameters = {
            "n_ctx": min(
                ml_predictions["n_ctx"], base_config["n_ctx"]
            ),  # Don't exceed profile limit
            "n_batch": min(ml_predictions["n_batch"], 2048),  # Reasonable upper bound
            "n_ubatch": min(
                ml_predictions["n_ubatch"], ml_predictions["n_batch"]
            ),  # n_ubatch ≤ n_batch
            "n_gpu_layers": min(
                ml_predictions["n_gpu_layers"], base_config["n_gpu_layers"]
            ),  # Don't exceed estimate
        }

        return optimized

    def execute_recovery_strategy(
        self,
        attempt: int,
        original_params: OptimalParameters,
        current_params: OptimalParameters,
        hardware_manager,
    ) -> RecoveryStrategy:
        """
        Execute OOM recovery strategy based on attempt number.

        Strategy levels (as requested):
        1-2: Clear memory only (hardware manager)
        3-4: Reduce batch/ubatch sizes
        5-6: Move layers to CPU (max 1/3 of total layers)
        7+: Reduce context size (last resort)

        Returns:
            RecoveryStrategy with new parameters and strategy name
        """
        new_params = current_params.copy()

        if attempt <= 2:
            # Level 1: Clear memory only, retry with same parameters
            strategy_name: Literal[
                "clear_memory", "reduce_batch", "move_to_cpu", "reduce_context"
            ] = "clear_memory"
            # Parameters stay the same, just clear memory via hardware manager

        elif attempt <= 4:
            # Level 2: Reduce batch/ubatch progressively
            strategy_name = "reduce_batch"
            reduction_factor = 2 ** (attempt - 2)  # 2x, 4x reduction
            new_params["n_batch"] = max(
                current_params["n_batch"] // reduction_factor, 32
            )
            new_params["n_ubatch"] = max(
                current_params["n_ubatch"] // reduction_factor, 32
            )

        elif attempt <= 6:
            # Level 3: Move layers to CPU (max 1/3 of total layers as specified)
            strategy_name = "move_to_cpu"
            original_gpu_layers = original_params["n_gpu_layers"]

            if original_gpu_layers > 0:
                # Calculate how many layers to move to CPU (max 1/3 as requested)
                max_cpu_layers = max(original_gpu_layers // 3, 1)  # At most 1/3 to CPU
                layers_to_move = min(
                    max_cpu_layers, (attempt - 4) * 5
                )  # Progressive movement
                new_params["n_gpu_layers"] = max(
                    original_gpu_layers - layers_to_move, 0
                )
            else:
                # If already CPU-only, reduce batch further
                new_params["n_batch"] = max(current_params["n_batch"] // 2, 16)
                new_params["n_ubatch"] = max(current_params["n_ubatch"] // 2, 16)

        else:
            # Level 4: Reduce context size (last resort as specified)
            strategy_name = "reduce_context"
            reduction_factor = 2 ** (attempt - 6)  # Progressive context reduction
            new_params["n_ctx"] = max(current_params["n_ctx"] // reduction_factor, 1024)

            # Also reduce batch sizes if context is very small
            if new_params["n_ctx"] <= 2048:
                new_params["n_batch"] = max(current_params["n_batch"] // 2, 16)
                new_params["n_ubatch"] = max(current_params["n_ubatch"] // 2, 16)

        result: RecoveryStrategy = {
            "parameters": new_params,
            "strategy_name": strategy_name,
        }

        self.logger.info(
            f"OOM recovery attempt {attempt}: strategy={strategy_name}, params={new_params}"
        )
        return result

    def record_success(
        self,
        model_path: str,
        params: OptimalParameters,
        hardware_manager,
        initialization_time_ms: float = 0.0,
        gpu_memory_used_mb: float = 0.0,
    ) -> None:
        """Record a successful configuration for ML training."""
        model_size_mb = self.get_model_size_mb(model_path)
        gpu_stats = self.get_system_gpu_stats(hardware_manager)

        config: ModelConfigurationData = {
            "n_ctx": params["n_ctx"],
            "n_batch": params["n_batch"],
            "n_ubatch": params["n_ubatch"],
            "n_gpu_layers": params["n_gpu_layers"],
            "model_size_mb": model_size_mb,
            "available_gpu_memory_mb": gpu_stats["available_memory"],
            "total_gpu_memory_mb": gpu_stats["total_memory"],
            "success": True,
            "gpu_memory_used_mb": (
                gpu_memory_used_mb if gpu_memory_used_mb > 0 else None
            ),
            "initialization_time_ms": (
                initialization_time_ms if initialization_time_ms > 0 else None
            ),
        }

        self.configurations.append(config)
        self._save_training_data()

        # Retrain models periodically (every 10 successful configurations)
        if len(self.configurations) % 10 == 0:
            self._train_models()

        self.logger.info(f"Recorded successful configuration: {params}")

    def record_failure(
        self,
        attempt: int,
        strategy: Literal[
            "clear_memory", "reduce_batch", "move_to_cpu", "reduce_context"
        ],
        params: OptimalParameters,
        error_message: str,
    ) -> None:
        """Record a failed recovery attempt."""
        recovery_attempt: OOMRecoveryAttemptData = {
            "attempt": attempt,
            "strategy": strategy,
            "n_ctx": params["n_ctx"],
            "n_batch": params["n_batch"],
            "n_ubatch": params["n_ubatch"],
            "n_gpu_layers": params["n_gpu_layers"],
            "success": False,
            "error_message": error_message,
        }

        self.recovery_attempts.append(recovery_attempt)
        self._save_training_data()

        self.logger.info(
            f"Recorded failed attempt {attempt}: {strategy}, error: {error_message[:100]}"
        )

    def _train_models(self) -> None:
        """Train Ridge regression models on collected data using sklearn (required)."""
        try:
            # Prepare training data from successful configurations
            feature_arrays: List[np.ndarray] = []
            targets: dict[str, List[int]] = {param: [] for param in self.models.keys()}

            for config in self.configurations:
                if config["success"]:  # TypedDict syntax
                    # Create GPU stats from stored data
                    gpu_stats: SystemGPUStats = {
                        "total_gpus": 1,  # Stored data may not have complete GPU info
                        "total_memory": config["total_gpu_memory_mb"],
                        "available_memory": config["available_gpu_memory_mb"],
                        "primary_gpu_id": 0,
                        "gpus": [],
                    }

                    # Extract features
                    features = self._extract_features(
                        config["model_size_mb"],
                        gpu_stats,
                        config["n_ctx"],
                        config["n_gpu_layers"],
                    )
                    feature_array = self._features_to_array(features)
                    feature_arrays.append(feature_array)

                    # Collect target values
                    targets["n_ctx"].append(config["n_ctx"])
                    targets["n_batch"].append(config["n_batch"])
                    targets["n_ubatch"].append(config["n_ubatch"])
                    targets["n_gpu_layers"].append(config["n_gpu_layers"])

            if len(feature_arrays) < 5:  # Need minimum data for training
                self.logger.info(
                    "Insufficient training data (need ≥5 samples), keeping current models"
                )
                return

            # Convert to numpy arrays
            features_matrix = np.array(feature_arrays)

            # Scale features
            features_scaled = self.scalers["features"].fit_transform(features_matrix)

            # Train Ridge regression model for each parameter (sklearn required)
            performance: dict[str, float] = {}

            for param_name in self.models.keys():
                target_values = np.array(targets[param_name])

                # Scale target values
                target_scaled = (
                    self.scalers[param_name]
                    .fit_transform(target_values.reshape(-1, 1))
                    .flatten()
                )

                # Use Ridge regression (sklearn required - no fallbacks)
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(features_scaled, target_scaled)
                self.models[param_name] = model

                # Calculate model performance
                pred_scaled = model.predict(features_scaled)
                mse = mean_squared_error(target_scaled, pred_scaled)
                performance[param_name] = mse

            self.logger.info(
                f"Trained ML models on {len(feature_arrays)} samples: "
                f"MSE scores: {performance}"
            )

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            # Note: No fallbacks - sklearn is required

    def _load_training_data(self) -> None:
        """Load training data from persistent storage."""
        try:
            config_file = self.data_dir / "configurations.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    data = json.load(f)
                    # Convert JSON data to properly typed configurations
                    typed_configs: List[ModelConfigurationData] = []
                    for item in data:
                        config: ModelConfigurationData = {
                            "n_ctx": item["n_ctx"],
                            "n_batch": item["n_batch"],
                            "n_ubatch": item["n_ubatch"],
                            "n_gpu_layers": item["n_gpu_layers"],
                            "model_size_mb": item["model_size_mb"],
                            "available_gpu_memory_mb": item["available_gpu_memory_mb"],
                            "total_gpu_memory_mb": item.get(
                                "total_gpu_memory_mb",
                                item.get("available_gpu_memory_mb", 0.0),
                            ),
                            "success": item["success"],
                            "gpu_memory_used_mb": item.get("gpu_memory_used_mb"),
                            "initialization_time_ms": item.get(
                                "initialization_time_ms"
                            ),
                        }
                        typed_configs.append(config)
                    self.configurations = typed_configs

            attempts_file = self.data_dir / "recovery_attempts.json"
            if attempts_file.exists():
                with open(attempts_file, "r") as f:
                    data = json.load(f)
                    # TypedDict creation from JSON data
                    self.recovery_attempts = data

            self.logger.info(
                f"Loaded {len(self.configurations)} configurations and {len(self.recovery_attempts)} recovery attempts"
            )

        except Exception as e:
            self.logger.warning(f"Error loading training data: {e}")

    def _save_training_data(self) -> None:
        """Save training data to persistent storage."""
        try:
            config_file = self.data_dir / "configurations.json"
            with open(config_file, "w") as f:
                json.dump(self.configurations, f, indent=2)

            attempts_file = self.data_dir / "recovery_attempts.json"
            with open(attempts_file, "w") as f:
                json.dump(self.recovery_attempts, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Error saving training data: {e}")

    def get_statistics(self) -> MLModelPerformance:
        """Get statistics about recovery performance with strong typing."""
        total_configs = len(self.configurations)
        successful_configs = sum(1 for c in self.configurations if c["success"])
        total_attempts = len(self.recovery_attempts)

        # Calculate strategy statistics with TypedDict syntax
        strategy_stats: dict[str, dict[str, int]] = {}
        for attempt in self.recovery_attempts:
            strategy = attempt["strategy"]
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "success": 0}
            strategy_stats[strategy]["total"] += 1
            if attempt["success"]:
                strategy_stats[strategy]["success"] += 1

        # Return strongly typed performance metrics
        performance: MLModelPerformance = {
            "n_ctx_mse": 0.0,
            "n_batch_mse": 0.0,
            "n_ubatch_mse": 0.0,
            "n_gpu_layers_mse": 0.0,
            "total_configurations": total_configs,
            "models_trained": all(model is not None for model in self.models.values()),
        }

        # Calculate MSE for trained models if available
        if performance["models_trained"] and total_configs > 0:
            try:
                # Re-evaluate model performance on current data
                feature_arrays = []
                targets = {param: [] for param in self.models.keys()}

                for config in self.configurations:
                    if config["success"]:
                        gpu_stats: SystemGPUStats = {
                            "total_gpus": 1,
                            "total_memory": config.get(
                                "total_gpu_memory_mb", config["available_gpu_memory_mb"]
                            ),
                            "available_memory": config["available_gpu_memory_mb"],
                            "primary_gpu_id": 0,
                            "gpus": [],
                        }

                        features = self._extract_features(
                            config["model_size_mb"],
                            gpu_stats,
                            config["n_ctx"],
                            config["n_gpu_layers"],
                        )
                        feature_arrays.append(self._features_to_array(features))

                        targets["n_ctx"].append(config["n_ctx"])
                        targets["n_batch"].append(config["n_batch"])
                        targets["n_ubatch"].append(config["n_ubatch"])
                        targets["n_gpu_layers"].append(config["n_gpu_layers"])

                if feature_arrays:
                    features_matrix = np.array(feature_arrays)
                    features_scaled = self.scalers["features"].transform(
                        features_matrix
                    )

                    for param_name in ["n_ctx", "n_batch", "n_ubatch", "n_gpu_layers"]:
                        model = self.models[param_name]
                        if model is not None:
                            target_values = np.array(targets[param_name])
                            target_scaled = (
                                self.scalers[param_name]
                                .transform(target_values.reshape(-1, 1))
                                .flatten()
                            )
                            pred_scaled = model.predict(features_scaled)
                            mse = mean_squared_error(target_scaled, pred_scaled)
                            performance[f"{param_name}_mse"] = mse  # type: ignore

            except Exception as e:
                self.logger.warning(f"Error calculating model performance: {e}")

        self.logger.info(
            f"Statistics: {total_configs} configs, {successful_configs} successful "
            f"({successful_configs/total_configs*100:.1f}% success rate), "
            f"{total_attempts} recovery attempts, models_trained={performance['models_trained']}"
        )

        return performance
