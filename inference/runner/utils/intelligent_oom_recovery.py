"""
Intelligent OOM Recovery System for LLM Pipeline Initialization.

Uses machine learning to predict optimal parameters and implements a structured
retry strategy for handling out-of-memory errors during model initialization.

Features strong typing with Pydantic models, dynamic multi-GPU support,
sklearn requirement, and model profile integration.
"""

import os
import json
import numpy as np
from typing import Optional, List, Literal
from pathlib import Path

# sklearn is required - no fallbacks
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from utils.logging import llmmllogger
from models.model_profile import ModelProfile
from models.dev_stats import DevStats
from models.system_gpu_stats import SystemGPUStats
from models.optimal_parameters import OptimalParameters
from models.model_configuration_data import ModelConfigurationData
from models.oom_recovery_attempt_data import OOMRecoveryAttemptData
from models.prediction_features import PredictionFeatures
from models.learned_limits import LearnedLimits
from models.recovery_strategy import RecoveryStrategy
from models.ml_model_performance import MLModelPerformance
from .hardware_manager import EnhancedHardwareManager


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
                    total_available_memory=0.0,
                    gpus=[],
                )

            # Collect GPU information using existing DevStats models
            gpus: List[DevStats] = []

            for _, stats in memory_stats.items():
                # Check if this is a GPU entry (DevStats with GPU-like attributes)
                if (
                    hasattr(stats, "mem_total")
                    and hasattr(stats, "mem_free")
                    and hasattr(stats, "name")
                    and "nvidia" in stats.name.lower()
                ):
                    # DevStats is already a Pydantic model, so we can use it directly
                    gpus.append(stats)

            if not gpus:
                # No GPUs found
                return SystemGPUStats(
                    total_gpus=0,
                    total_memory=0.0,
                    total_available_memory=0.0,
                    gpus=[],
                )

            # Sort GPUs by ID for consistent ordering
            gpus.sort(key=lambda x: int(x.id) if x.id.isdigit() else 0)

            # Calculate system totals
            total_memory = sum(gpu.mem_total for gpu in gpus)
            total_available_memory = sum(gpu.mem_free for gpu in gpus)

            system_stats = SystemGPUStats(
                total_gpus=len(gpus),
                total_memory=total_memory,
                total_available_memory=total_available_memory,
                gpus=gpus,
            )

            # Log comprehensive GPU information
            gpu_list = ", ".join(
                [
                    f"GPU{gpu.id}({gpu.mem_free:.0f}/{gpu.mem_total:.0f}MB)"
                    for gpu in gpus
                ]
            )

            self.logger.info(
                f"Multi-GPU System: {len(gpus)} GPUs, Total: {total_memory:.0f}MB, "
                f"Total Available: {total_available_memory:.0f}MB, GPUs: [{gpu_list}]"
            )

            return system_stats

        except Exception as e:
            self.logger.warning(f"Error getting GPU system stats: {e}")
            return SystemGPUStats(
                total_gpus=0,
                total_memory=0.0,
                total_available_memory=0.0,
                gpus=[],
            )

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

        # Apply learned constraints based on historical success data
        learned_limits = self._get_learned_limits()

        optimized_config = OptimalParameters(
            n_ctx=min(base_n_ctx, learned_limits.max_context),
            n_batch=min(base_batch_size, learned_limits.max_batch),
            n_ubatch=min(base_n_ubatch, base_batch_size),
            n_gpu_layers=min(base_n_gpu_layers, learned_limits.max_gpu_layers),
        )

        self.logger.info(
            f"Profile-driven config: n_ctx={optimized_config.n_ctx}, "
            f"n_batch={optimized_config.n_batch}, n_ubatch={optimized_config.n_ubatch}, "
            f"n_gpu_layers={optimized_config.n_gpu_layers} (from {model_profile.name})"
        )

        return optimized_config

    def _get_learned_minimums(self) -> OptimalParameters:
        """Get learned minimum values from successful configurations."""
        if len(self.configurations) < 5:
            # Insufficient data, use conservative initial minimums
            return OptimalParameters(
                n_ctx=1024,  # Conservative minimum context
                n_batch=16,  # Conservative minimum batch
                n_ubatch=16,  # Conservative minimum ubatch
                n_gpu_layers=0,  # Minimum GPU layers (CPU fallback)
            )

        # Extract successful configurations for safe parameter estimation
        contexts = [config.n_ctx for config in self.configurations if config.success]
        batches = [config.n_batch for config in self.configurations if config.success]
        ubatches = [config.n_ubatch for config in self.configurations if config.success]
        gpu_layers = [
            config.n_gpu_layers for config in self.configurations if config.success
        ]

        # Use 10th percentile as safe minimum (tested values that worked)
        contexts_sorted = sorted(contexts)
        batches_sorted = sorted(batches)
        ubatches_sorted = sorted(ubatches)
        gpu_layers_sorted = sorted(gpu_layers)

        p10_idx = max(0, int(len(contexts) * 0.1))

        return OptimalParameters(
            n_ctx=max(contexts_sorted[p10_idx], 512),  # Never go below 512 context
            n_batch=max(batches_sorted[p10_idx], 8),  # Never go below 8 batch
            n_ubatch=max(ubatches_sorted[p10_idx], 8),  # Never go below 8 ubatch
            n_gpu_layers=gpu_layers_sorted[p10_idx],  # Can be 0 for CPU-only
        )

    def _get_learned_limits(self) -> LearnedLimits:
        """Get learned maximum limits from successful configurations."""
        if len(self.configurations) < 5:
            # Insufficient data for learning, use conservative initial estimates
            return LearnedLimits(
                max_context=16384,  # Conservative initial estimate
                max_batch=256,  # Conservative initial estimate
                max_gpu_layers=64,  # Conservative initial estimate
                context_95th_percentile=8192,
                batch_95th_percentile=128,
                gpu_layers_95th_percentile=32,
            )

        # Extract parameters from successful configurations
        contexts = [config.n_ctx for config in self.configurations]
        batches = [config.n_batch for config in self.configurations]
        gpu_layers = [config.n_gpu_layers for config in self.configurations]

        # Calculate statistics
        max_context = max(contexts)
        max_batch = max(batches)
        max_gpu_layers_val = max(gpu_layers)

        # Calculate 95th percentiles for safer limits
        contexts_sorted = sorted(contexts)
        batches_sorted = sorted(batches)
        gpu_layers_sorted = sorted(gpu_layers)

        p95_idx = int(len(contexts) * 0.95)
        context_95th = contexts_sorted[min(p95_idx, len(contexts) - 1)]
        batch_95th = batches_sorted[min(p95_idx, len(batches) - 1)]
        gpu_layers_95th = gpu_layers_sorted[min(p95_idx, len(gpu_layers) - 1)]

        return LearnedLimits(
            max_context=max_context,
            max_batch=max_batch,
            max_gpu_layers=max_gpu_layers_val,
            context_95th_percentile=context_95th,
            batch_95th_percentile=batch_95th,
            gpu_layers_95th_percentile=gpu_layers_95th,
        )

    def _estimate_gpu_layers_from_system(self, gpu_stats: SystemGPUStats) -> int:
        """Estimate optimal GPU layers based on system capabilities and learned data."""
        if gpu_stats.total_gpus == 0:
            return 0  # CPU-only

        learned_limits = self._get_learned_limits()

        # Use learned data if available, otherwise conservative estimate
        if len(self.configurations) >= 5:
            # Use 95th percentile as safe upper bound
            return min(
                learned_limits.gpu_layers_95th_percentile,
                learned_limits.max_gpu_layers,
            )
        else:
            # Initial conservative estimate based on total memory
            total_memory_gb = gpu_stats.total_memory / 1024
            estimated_layers = int(
                total_memory_gb * 4
            )  # ~4 layers per GB as initial estimate
            return max(0, min(estimated_layers, 64))

    def _extract_features(
        self,
        model_size_mb: float,
        gpu_stats: SystemGPUStats,
    ) -> PredictionFeatures:
        """
        Extract features for ML prediction with comprehensive multi-GPU awareness.

        Returns structured features optimized for Ridge regression models.
        """
        total_available_memory = gpu_stats.total_available_memory
        total_gpu_memory = gpu_stats.total_memory
        gpu_count = gpu_stats.total_gpus

        features = PredictionFeatures(
            model_size_mb=model_size_mb,
            total_gpu_memory_mb=total_gpu_memory,
            total_available_memory_mb=total_available_memory,
            total_gpus=gpu_count,
        )

        return features

    def _features_to_array(self, features: PredictionFeatures) -> np.ndarray:
        """Convert feature dictionary to numpy array for sklearn."""
        return np.array(
            [
                features.model_size_mb,
                features.total_gpu_memory_mb,
                features.total_available_memory_mb,
                features.total_gpus,
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
                ml_predictions[param_name] = getattr(base_config, param_name)

        # Combine ML predictions with profile constraints
        optimized = OptimalParameters(
            n_ctx=min(
                ml_predictions["n_ctx"],
                base_config.n_ctx,
            ),  # Don't exceed profile limit
            n_batch=min(ml_predictions["n_batch"], 2048),  # Reasonable upper bound
            n_ubatch=min(
                ml_predictions["n_ubatch"],
                ml_predictions["n_batch"],
            ),  # n_ubatch ≤ n_batch
            n_gpu_layers=min(
                ml_predictions["n_gpu_layers"],
                base_config.n_gpu_layers,
            ),  # Don't exceed estimate
        )

        return optimized

    def execute_recovery_strategy(
        self,
        attempt: int,
        original_params: OptimalParameters,
        current_params: OptimalParameters,
        hardware_manager: EnhancedHardwareManager,  # noqa: ARG002  # Reserved for future memory clearing
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
        if attempt <= 2:
            # Level 1: Clear memory only, retry with same parameters
            strategy_name: Literal[
                "clear_memory", "reduce_batch", "move_to_cpu", "reduce_context"
            ] = "clear_memory"
            new_params = current_params
            # Parameters stay the same, just clear memory via hardware manager
            if attempt == 0:
                hardware_manager.clear_memory()
            elif attempt == 1:
                hardware_manager.clear_memory(aggressive=True)
            else:
                hardware_manager.clear_memory(aggressive=True, nuclear=True)

        elif attempt <= 4:
            # Level 2: Reduce batch/ubatch progressively
            strategy_name = "reduce_batch"
            reduction_factor = 2 ** (attempt - 2)  # 2x, 4x reduction
            new_params = current_params.model_copy(
                update={
                    "n_batch": max(current_params.n_batch // reduction_factor, 32),
                    "n_ubatch": max(current_params.n_ubatch // reduction_factor, 32),
                }
            )

        elif attempt <= 6:
            # Level 3: Move layers to CPU (max 1/3 of total layers as specified)
            strategy_name = "move_to_cpu"
            original_gpu_layers = original_params.n_gpu_layers

            if original_gpu_layers > 0:
                # Calculate how many layers to move to CPU (max 1/3 as requested)
                max_cpu_layers = max(original_gpu_layers // 3, 1)  # At most 1/3 to CPU
                layers_to_move = min(
                    max_cpu_layers, (attempt - 4) * 5
                )  # Progressive movement
                new_params = current_params.model_copy(
                    update={
                        "n_gpu_layers": max(original_gpu_layers - layers_to_move, 0)
                    }
                )
            else:
                # If already CPU-only, reduce batch further
                learned_mins = self._get_learned_minimums()
                new_params = current_params.model_copy(
                    update={
                        "n_batch": max(
                            current_params.n_batch // 2, learned_mins.n_batch
                        ),
                        "n_ubatch": max(
                            current_params.n_ubatch // 2, learned_mins.n_ubatch
                        ),
                    }
                )

        else:
            # Level 4: Reduce context size (last resort as specified)
            strategy_name = "reduce_context"
            learned_mins = self._get_learned_minimums()
            reduction_factor = 2 ** (attempt - 6)  # Progressive context reduction
            new_ctx = max(current_params.n_ctx // reduction_factor, learned_mins.n_ctx)

            updates = {"n_ctx": new_ctx}
            # Also reduce batch sizes if context is very small
            if new_ctx <= learned_mins.n_ctx * 2:
                updates.update(
                    {
                        "n_batch": max(
                            current_params.n_batch // 2, learned_mins.n_batch
                        ),
                        "n_ubatch": max(
                            current_params.n_ubatch // 2, learned_mins.n_ubatch
                        ),
                    }
                )

            new_params = current_params.model_copy(update=updates)

        result = RecoveryStrategy(
            parameters=new_params,
            strategy_name=strategy_name,
        )

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

        config = ModelConfigurationData(
            n_ctx=params.n_ctx,
            n_batch=params.n_batch,
            n_ubatch=params.n_ubatch,
            n_gpu_layers=params.n_gpu_layers,
            model_size_mb=model_size_mb,
            available_gpu_memory_mb=gpu_stats.total_available_memory,
            total_gpu_memory_mb=gpu_stats.total_memory,
            success=True,
            gpu_memory_used_mb=(gpu_memory_used_mb if gpu_memory_used_mb > 0 else None),
            initialization_time_ms=(
                initialization_time_ms if initialization_time_ms > 0 else None
            ),
        )

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
        recovery_attempt = OOMRecoveryAttemptData(
            attempt=attempt,
            strategy=strategy,
            n_ctx=params.n_ctx,
            n_batch=params.n_batch,
            n_ubatch=params.n_ubatch,
            n_gpu_layers=params.n_gpu_layers,
            success=False,
            error_message=error_message,
        )

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
                if config.success:
                    # Create GPU stats from stored data
                    gpu_stats = SystemGPUStats(
                        total_gpus=1,  # Stored data may not have complete GPU info
                        total_memory=config.total_gpu_memory_mb,
                        total_available_memory=config.available_gpu_memory_mb,
                        gpus=[],
                    )

                    # Extract features
                    features = self._extract_features(
                        config.model_size_mb,
                        gpu_stats,
                    )
                    feature_array = self._features_to_array(features)
                    feature_arrays.append(feature_array)

                    # Collect target values
                    targets["n_ctx"].append(config.n_ctx)
                    targets["n_batch"].append(config.n_batch)
                    targets["n_ubatch"].append(config.n_ubatch)
                    targets["n_gpu_layers"].append(config.n_gpu_layers)

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
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert JSON data to properly typed configurations
                    typed_configs: List[ModelConfigurationData] = []
                    for item in data:
                        config = ModelConfigurationData(
                            n_ctx=item["n_ctx"],
                            n_batch=item["n_batch"],
                            n_ubatch=item["n_ubatch"],
                            n_gpu_layers=item["n_gpu_layers"],
                            model_size_mb=item["model_size_mb"],
                            available_gpu_memory_mb=item["available_gpu_memory_mb"],
                            total_gpu_memory_mb=item.get(
                                "total_gpu_memory_mb",
                                item.get("available_gpu_memory_mb", 0.0),
                            ),
                            success=item["success"],
                            gpu_memory_used_mb=item.get("gpu_memory_used_mb"),
                            initialization_time_ms=item.get("initialization_time_ms"),
                        )
                        typed_configs.append(config)
                    self.configurations = typed_configs

            attempts_file = self.data_dir / "recovery_attempts.json"
            if attempts_file.exists():
                with open(attempts_file, "r", encoding="utf-8") as f:
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
            with open(config_file, "w", encoding="utf-8") as f:
                # Convert Pydantic models to dicts for JSON serialization
                config_dicts = [config.model_dump() for config in self.configurations]
                json.dump(config_dicts, f, indent=2)

            attempts_file = self.data_dir / "recovery_attempts.json"
            with open(attempts_file, "w", encoding="utf-8") as f:
                # Convert Pydantic models to dicts for JSON serialization
                attempt_dicts = [
                    attempt.model_dump() for attempt in self.recovery_attempts
                ]
                json.dump(attempt_dicts, f, indent=2)

        except Exception as e:
            self.logger.warning(f"Error saving training data: {e}")

    def get_statistics(self) -> MLModelPerformance:
        """Get statistics about recovery performance with strong typing."""
        total_configs = len(self.configurations)
        successful_configs = sum(1 for c in self.configurations if c.success)
        total_attempts = len(self.recovery_attempts)

        # Calculate strategy statistics with TypedDict syntax
        strategy_stats: dict[str, dict[str, int]] = {}
        for attempt in self.recovery_attempts:
            strategy = attempt.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "success": 0}
            strategy_stats[strategy]["total"] += 1
            if attempt.success:
                strategy_stats[strategy]["success"] += 1

        # Return strongly typed performance metrics
        performance = MLModelPerformance(
            n_ctx_mse=0.0,
            n_batch_mse=0.0,
            n_ubatch_mse=0.0,
            n_gpu_layers_mse=0.0,
            total_configurations=total_configs,
            models_trained=all(model is not None for model in self.models.values()),
        )

        # Calculate MSE for trained models if available
        if performance.models_trained and total_configs > 0:
            try:
                # Re-evaluate model performance on current data
                feature_arrays = []
                targets = {param: [] for param in self.models.keys()}

                for config in self.configurations:
                    if config.success:
                        gpu_stats = SystemGPUStats(
                            total_gpus=1,
                            total_memory=getattr(
                                config,
                                "total_gpu_memory_mb",
                                config.available_gpu_memory_mb,
                            ),
                            total_available_memory=config.available_gpu_memory_mb,
                            gpus=[],
                        )

                        features = self._extract_features(
                            config.model_size_mb,
                            gpu_stats,
                        )
                        feature_arrays.append(self._features_to_array(features))

                        targets["n_ctx"].append(config.n_ctx)
                        targets["n_batch"].append(config.n_batch)
                        targets["n_ubatch"].append(config.n_ubatch)
                        targets["n_gpu_layers"].append(config.n_gpu_layers)

                if feature_arrays:
                    features_matrix = np.array(feature_arrays)
                    features_scaled = self.scalers["features"].transform(
                        features_matrix
                    )

                    mse_values = {
                        "n_ctx_mse": 0.0,
                        "n_batch_mse": 0.0,
                        "n_ubatch_mse": 0.0,
                        "n_gpu_layers_mse": 0.0,
                    }

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
                            mse_values[f"{param_name}_mse"] = mse

                    performance = performance.model_copy(update=mse_values)

            except Exception as e:
                self.logger.warning(f"Error calculating model performance: {e}")

        self.logger.info(
            f"Statistics: {total_configs} configs, {successful_configs} successful "
            f"({successful_configs/total_configs*100:.1f}% success rate), "
            f"{total_attempts} recovery attempts, models_trained={performance.models_trained}"
        )

        return performance
