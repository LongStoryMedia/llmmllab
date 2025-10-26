"""
Intelligent OOM Recovery System for LLM Pipeline Initialization.

Uses machine learning to predict optimal parameters and implements a structured
retry strategy for handling out-of-memory errors during model initialization.
"""

import os
import json
import hashlib
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from utils.logging import llmmllogger


@dataclass
class ModelConfiguration:
    """Configuration parameters for model initialization."""

    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    model_size_mb: float
    available_gpu_memory_mb: float
    success: bool
    gpu_memory_used_mb: Optional[float] = None
    initialization_time_ms: Optional[float] = None


@dataclass
class OOMRecoveryAttempt:
    """Single OOM recovery attempt configuration."""

    attempt: int
    strategy: str  # "clear_memory", "reduce_batch", "move_to_cpu", "reduce_context"
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    success: bool
    error_message: str = ""


class IntelligentOOMRecovery:
    """
    Intelligent OOM recovery system using machine learning to predict optimal parameters.

    Features:
    - ML-based parameter prediction using model size and system resources
    - Structured retry strategy with 4 levels: clear memory -> reduce batch -> move to CPU -> reduce context
    - Learning from successful configurations to improve future predictions
    - Persistent storage of training data for cross-session learning
    """

    def __init__(self, data_dir: str = "/tmp/oom_recovery_data"):
        self.logger = llmmllogger.bind(component="IntelligentOOMRecovery")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ML models for predicting optimal parameters
        self.models = {
            "n_ctx": None,
            "n_batch": None,
            "n_ubatch": None,
            "n_gpu_layers": None,
        }
        self.scalers = {
            "features": StandardScaler(),
            "n_ctx": StandardScaler(),
            "n_batch": StandardScaler(),
            "n_ubatch": StandardScaler(),
            "n_gpu_layers": StandardScaler(),
        }

        # Training data storage
        self.configurations: List[ModelConfiguration] = []
        self.recovery_attempts: List[OOMRecoveryAttempt] = []

        # Load existing data and train models
        self._load_training_data()
        self._train_models()

    def get_model_size_mb(self, model_path: str) -> float:
        """Get model file size in MB."""
        try:
            if os.path.exists(model_path):
                size_bytes = os.path.getsize(model_path)
                return size_bytes / (1024 * 1024)
            else:
                # Fallback: estimate based on filename patterns
                if "7b" in model_path.lower():
                    return 4000  # ~4GB
                elif "13b" in model_path.lower():
                    return 8000  # ~8GB
                elif "30b" in model_path.lower() or "33b" in model_path.lower():
                    return 20000  # ~20GB
                elif "70b" in model_path.lower():
                    return 40000  # ~40GB
                else:
                    return 8000  # Default estimate
        except Exception as e:
            self.logger.warning(f"Error getting model size: {e}")
            return 8000  # Default fallback

    def get_available_gpu_memory_mb(self, hardware_manager) -> float:
        """
        Get available GPU memory in MB from the primary GPU.

        For multiple GPU systems, prioritizes:
        1. GPU with id=0 (primary GPU, cuda:0)
        2. GPU with most available memory
        3. First GPU found
        """
        try:
            memory_stats = hardware_manager.update_all_memory_stats()
            if not memory_stats:
                return 8000  # Fallback estimate

            # Collect all GPU stats for analysis
            gpu_candidates = []

            for key, stats in memory_stats.items():
                # Check if this is a GPU entry (DevStats with GPU-like attributes)
                if (
                    hasattr(stats, "mem_total")
                    and hasattr(stats, "mem_free")
                    and hasattr(stats, "name")
                    and "nvidia" in stats.name.lower()
                ):

                    gpu_info = {
                        "key": key,
                        "stats": stats,
                        "id": getattr(stats, "id", key),
                        "name": getattr(stats, "name", f"GPU {key}"),
                        "available_memory": max(getattr(stats, "mem_free", 0), 0),
                        "total_memory": max(getattr(stats, "mem_total", 0), 0),
                        "used_memory": max(getattr(stats, "mem_used", 0), 0),
                    }

                    gpu_candidates.append(gpu_info)

            if not gpu_candidates:
                return 8000  # No GPUs found, use fallback

            # Intelligent GPU selection strategy
            selected_gpu = None
            gpu_0 = None

            # Find GPU 0 if it exists
            for gpu in gpu_candidates:
                if gpu["id"] == 0:
                    gpu_0 = gpu
                    break

            # Selection logic: intelligent GPU selection with consistency preference
            if gpu_0 is not None:
                if len(gpu_candidates) == 1:
                    # Only one GPU available
                    selected_gpu = gpu_0
                elif gpu_0["available_memory"] < 8000:
                    # GPU 0 has insufficient memory, use the best available
                    selected_gpu = max(
                        gpu_candidates, key=lambda x: x["available_memory"]
                    )
                else:
                    # GPU 0 has sufficient memory (≥8GB), but check if another GPU is significantly better
                    best_gpu = max(gpu_candidates, key=lambda x: x["available_memory"])

                    # Only switch if best GPU has substantially more memory (>2x) AND is not GPU 0
                    if (
                        best_gpu["id"] != gpu_0["id"]
                        and best_gpu["available_memory"] > gpu_0["available_memory"] * 2
                    ):
                        selected_gpu = best_gpu
                        self.logger.info(
                            f"Switching from GPU 0 to GPU {best_gpu['id']} due to significantly more memory "
                            f"({best_gpu['available_memory']} vs {gpu_0['available_memory']} MB)"
                        )
                    else:
                        selected_gpu = gpu_0  # Prefer GPU 0 for consistency
                        self.logger.info(
                            f"Using GPU 0 for consistency despite other options available"
                        )
            else:
                # No GPU 0 found, use the one with most available memory
                selected_gpu = max(gpu_candidates, key=lambda x: x["available_memory"])

            # Final fallback
            if selected_gpu is None:
                selected_gpu = gpu_candidates[0]

            available_memory = selected_gpu["available_memory"]

            # Log GPU selection info for debugging
            self.logger.info(
                f"Selected GPU {selected_gpu['id']} ({selected_gpu['name']}): "
                f"{available_memory:.0f} MB available / {selected_gpu['total_memory']:.0f} MB total "
                f"(from {len(gpu_candidates)} GPUs)"
            )

            # Return available memory with reasonable minimum
            return max(available_memory, 1000) if available_memory > 0 else 8000

        except Exception as e:
            self.logger.warning(f"Error getting GPU memory: {e}")
            return 8000  # Fallback estimate

    def get_total_gpu_memory_mb(self, hardware_manager) -> float:
        """Get total GPU memory across all GPUs in the system."""
        try:
            memory_stats = hardware_manager.update_all_memory_stats()
            if not memory_stats:
                return 8000  # Fallback estimate

            total_memory = 0
            gpu_count = 0
            gpu_details = []

            for key, stats in memory_stats.items():
                # Check if this is a GPU entry (DevStats with GPU-like attributes)
                if (
                    hasattr(stats, "mem_total")
                    and hasattr(stats, "name")
                    and "nvidia" in stats.name.lower()
                ):

                    gpu_memory = getattr(stats, "mem_total", 0)
                    total_memory += gpu_memory
                    gpu_count += 1

                    gpu_details.append(
                        {
                            "id": getattr(stats, "id", key),
                            "name": getattr(stats, "name", f"GPU {key}"),
                            "memory": gpu_memory,
                        }
                    )

            if total_memory > 0:
                # Log detailed GPU information
                gpu_list = ", ".join(
                    [f"GPU{gpu['id']}({gpu['memory']:.0f}MB)" for gpu in gpu_details]
                )
                self.logger.info(
                    f"Total GPU memory: {total_memory:.0f} MB across {gpu_count} GPUs [{gpu_list}]"
                )
                return total_memory
            else:
                return 8000  # Fallback estimate

        except Exception as e:
            self.logger.warning(f"Error getting total GPU memory: {e}")
            return 8000  # Fallback estimate

    def _extract_features(
        self,
        model_size_mb: float,
        target_n_ctx: int,
        available_gpu_memory_mb: float,
        requested_gpu_layers: int,
        hardware_manager=None,
    ) -> np.ndarray:
        """Extract features for ML prediction, including multi-GPU awareness."""

        # Get total GPU memory for better system understanding
        total_gpu_memory_mb = available_gpu_memory_mb  # Default fallback
        if hardware_manager:
            total_gpu_memory_mb = self.get_total_gpu_memory_mb(hardware_manager)

        return np.array(
            [
                model_size_mb,
                target_n_ctx,
                available_gpu_memory_mb,  # Primary GPU memory
                requested_gpu_layers,
                model_size_mb
                / available_gpu_memory_mb,  # Memory pressure ratio (primary GPU)
                target_n_ctx / 1000,  # Context size in thousands
                np.log(model_size_mb + 1),  # Log model size for non-linear relationship
                total_gpu_memory_mb,  # Total system GPU memory
                model_size_mb / total_gpu_memory_mb,  # System-wide memory pressure
                available_gpu_memory_mb
                / total_gpu_memory_mb,  # Primary GPU memory fraction
            ]
        )

    def predict_optimal_parameters(
        self,
        model_path: str,
        target_n_ctx: int,
        target_batch: int,
        target_ubatch: int,
        requested_gpu_layers: int,
        hardware_manager,
    ) -> Dict[str, int]:
        """
        Predict optimal parameters using ML models or heuristics.

        Returns:
            Dict with predicted n_ctx, n_batch, n_ubatch, n_gpu_layers
        """
        model_size_mb = self.get_model_size_mb(model_path)
        available_gpu_memory_mb = self.get_available_gpu_memory_mb(hardware_manager)

        features = self._extract_features(
            model_size_mb,
            target_n_ctx,
            available_gpu_memory_mb,
            requested_gpu_layers,
            hardware_manager,
        )

        # Use ML models for prediction
        features_scaled = self.scalers["features"].transform([features])

        predictions = {}
        for param_name, model in self.models.items():
            if model is not None:
                pred_scaled = model.predict(features_scaled)[0]
                pred_value = self.scalers[param_name].inverse_transform(
                    [[pred_scaled]]
                )[0][0]
                predictions[param_name] = max(int(pred_value), 1)
            else:
                predictions[param_name] = self._get_fallback_value(
                    param_name,
                    target_n_ctx,
                    target_batch,
                    target_ubatch,
                    requested_gpu_layers,
                )

                self.logger.info(f"ML predicted parameters: {predictions}")
                return predictions

        # Fallback to heuristic-based prediction
        return self._heuristic_prediction(
            model_size_mb,
            target_n_ctx,
            target_batch,
            target_ubatch,
            requested_gpu_layers,
            available_gpu_memory_mb,
        )

    def _get_fallback_value(
        self,
        param_name: str,
        target_n_ctx: int,
        target_batch: int,
        target_ubatch: int,
        requested_gpu_layers: int,
    ) -> int:
        """Get fallback value for a parameter."""
        fallbacks = {
            "n_ctx": target_n_ctx,
            "n_batch": target_batch,
            "n_ubatch": target_ubatch,
            "n_gpu_layers": requested_gpu_layers,
        }
        return fallbacks.get(param_name, 512)

    def _heuristic_prediction(
        self,
        model_size_mb: float,
        target_n_ctx: int,
        target_batch: int,
        target_ubatch: int,
        requested_gpu_layers: int,
        available_gpu_memory_mb: float,
    ) -> Dict[str, int]:
        """Heuristic-based parameter prediction when ML is not available."""

        # Calculate memory pressure
        memory_pressure = model_size_mb / available_gpu_memory_mb

        # Conservative adjustments based on memory pressure
        if memory_pressure > 0.8:  # High memory pressure
            n_ctx = max(target_n_ctx // 2, 2048)
            n_batch = max(target_batch // 4, 64)
            n_ubatch = max(target_ubatch // 4, 64)
            n_gpu_layers = max(requested_gpu_layers // 2, 0)
        elif memory_pressure > 0.6:  # Medium memory pressure
            n_ctx = max(int(target_n_ctx * 0.75), 4096)
            n_batch = max(target_batch // 2, 128)
            n_ubatch = max(target_ubatch // 2, 128)
            n_gpu_layers = max(int(requested_gpu_layers * 0.8), 0)
        else:  # Low memory pressure - use targets
            n_ctx = target_n_ctx
            n_batch = target_batch
            n_ubatch = target_ubatch
            n_gpu_layers = requested_gpu_layers

        return {
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "n_gpu_layers": n_gpu_layers,
        }

    def execute_recovery_strategy(
        self,
        attempt: int,
        original_params: Dict[str, int],
        current_params: Dict[str, int],
        hardware_manager,
    ) -> Tuple[Dict[str, int], str]:
        """
        Execute OOM recovery strategy based on attempt number.

        Strategy levels:
        1-2: Clear memory only
        3-4: Reduce batch/ubatch sizes
        5-6: Move layers to CPU (max 1/3)
        7+: Reduce context size

        Returns:
            Tuple of (new_parameters, strategy_name)
        """

        strategy = ""
        new_params = current_params.copy()

        if attempt <= 2:
            # Level 1: Clear memory only, retry with same parameters
            strategy = "clear_memory"
            # Parameters stay the same, just clear memory

        elif attempt <= 4:
            # Level 2: Reduce batch/ubatch progressively
            strategy = "reduce_batch"
            reduction_factor = 2 ** (attempt - 2)  # 2x, 4x reduction
            new_params["n_batch"] = max(
                current_params["n_batch"] // reduction_factor, 32
            )
            new_params["n_ubatch"] = max(
                current_params["n_ubatch"] // reduction_factor, 32
            )

        elif attempt <= 6:
            # Level 3: Move layers to CPU (max 1/3 of total layers)
            strategy = "move_to_cpu"
            original_gpu_layers = original_params["n_gpu_layers"]

            if original_gpu_layers > 0:
                # Calculate how many layers to move to CPU
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
            # Level 4: Reduce context size (last resort)
            strategy = "reduce_context"
            reduction_factor = 2 ** (attempt - 6)  # Progressive context reduction
            new_params["n_ctx"] = max(current_params["n_ctx"] // reduction_factor, 1024)

            # Also reduce batch sizes if context is very small
            if new_params["n_ctx"] <= 2048:
                new_params["n_batch"] = max(current_params["n_batch"] // 2, 16)
                new_params["n_ubatch"] = max(current_params["n_ubatch"] // 2, 16)

        self.logger.info(
            f"OOM recovery attempt {attempt}: strategy={strategy}, params={new_params}"
        )
        return new_params, strategy

    def record_success(
        self,
        model_path: str,
        params: Dict[str, int],
        hardware_manager,
        initialization_time_ms: float = 0,
        gpu_memory_used_mb: float = 0,
    ):
        """Record a successful configuration for ML training."""

        model_size_mb = self.get_model_size_mb(model_path)
        available_gpu_memory_mb = self.get_available_gpu_memory_mb(hardware_manager)

        config = ModelConfiguration(
            n_ctx=params["n_ctx"],
            n_batch=params["n_batch"],
            n_ubatch=params["n_ubatch"],
            n_gpu_layers=params["n_gpu_layers"],
            model_size_mb=model_size_mb,
            available_gpu_memory_mb=available_gpu_memory_mb,
            success=True,
            gpu_memory_used_mb=gpu_memory_used_mb,
            initialization_time_ms=initialization_time_ms,
        )

        self.configurations.append(config)
        self._save_training_data()

        # Retrain models periodically
        if len(self.configurations) % 10 == 0:
            self._train_models()

    def record_failure(
        self, attempt: int, strategy: str, params: Dict[str, int], error_message: str
    ):
        """Record a failed recovery attempt."""

        recovery_attempt = OOMRecoveryAttempt(
            attempt=attempt,
            strategy=strategy,
            n_ctx=params["n_ctx"],
            n_batch=params["n_batch"],
            n_ubatch=params["n_ubatch"],
            n_gpu_layers=params["n_gpu_layers"],
            success=False,
            error_message=error_message,
        )

        self.recovery_attempts.append(recovery_attempt)
        self._save_training_data()

    def _train_models(self):
        """Train ML models on collected data."""
        try:
            # Prepare training data from successful configurations
            features = []
            targets = {param: [] for param in self.models.keys()}

            for config in self.configurations:
                if config.success:
                    feature_vector = self._extract_features(
                        config.model_size_mb,
                        config.n_ctx,
                        config.available_gpu_memory_mb,
                        config.n_gpu_layers,
                    )
                    features.append(feature_vector)
                    targets["n_ctx"].append(config.n_ctx)
                    targets["n_batch"].append(config.n_batch)
                    targets["n_ubatch"].append(config.n_ubatch)
                    targets["n_gpu_layers"].append(config.n_gpu_layers)

            if len(features) < 5:  # Need minimum data for training
                return

            features = np.array(features)

            # Scale features
            features_scaled = self.scalers["features"].fit_transform(features)

            # Train a model for each parameter
            for param_name in self.models.keys():
                target_values = np.array(targets[param_name])

                # Scale target values
                target_scaled = (
                    self.scalers[param_name]
                    .fit_transform(target_values.reshape(-1, 1))
                    .flatten()
                )

                # Use Ridge regression for better generalization
                model = Ridge(alpha=1.0)
                model.fit(features_scaled, target_scaled)
                self.models[param_name] = model

                # Calculate and log model performance
                pred_scaled = model.predict(features_scaled)
                mse = mean_squared_error(target_scaled, pred_scaled)
                self.logger.info(f"Trained {param_name} model: MSE={mse:.4f}")

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")

    def _load_training_data(self):
        """Load training data from persistent storage."""
        try:
            config_file = self.data_dir / "configurations.json"
            if config_file.exists():
                with open(config_file, "r") as f:
                    data = json.load(f)
                    self.configurations = [ModelConfiguration(**item) for item in data]

            attempts_file = self.data_dir / "recovery_attempts.json"
            if attempts_file.exists():
                with open(attempts_file, "r") as f:
                    data = json.load(f)
                    self.recovery_attempts = [
                        OOMRecoveryAttempt(**item) for item in data
                    ]

            self.logger.info(
                f"Loaded {len(self.configurations)} configurations and {len(self.recovery_attempts)} recovery attempts"
            )

        except Exception as e:
            self.logger.warning(f"Error loading training data: {e}")

    def _save_training_data(self):
        """Save training data to persistent storage."""
        try:
            config_file = self.data_dir / "configurations.json"
            with open(config_file, "w") as f:
                json.dump(
                    [asdict(config) for config in self.configurations], f, indent=2
                )

            attempts_file = self.data_dir / "recovery_attempts.json"
            with open(attempts_file, "w") as f:
                json.dump(
                    [asdict(attempt) for attempt in self.recovery_attempts], f, indent=2
                )

        except Exception as e:
            self.logger.warning(f"Error saving training data: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery performance."""
        total_configs = len(self.configurations)
        successful_configs = sum(1 for c in self.configurations if c.success)
        total_attempts = len(self.recovery_attempts)

        strategy_stats = {}
        for attempt in self.recovery_attempts:
            strategy = attempt.strategy
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {"total": 0, "success": 0}
            strategy_stats[strategy]["total"] += 1
            if attempt.success:
                strategy_stats[strategy]["success"] += 1

        return {
            "total_configurations": total_configs,
            "successful_configurations": successful_configs,
            "success_rate": (
                successful_configs / total_configs if total_configs > 0 else 0
            ),
            "total_recovery_attempts": total_attempts,
            "strategy_statistics": strategy_stats,
            "ml_models_trained": all(
                model is not None for model in self.models.values()
            ),
        }
