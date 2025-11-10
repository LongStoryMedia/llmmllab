"""
LlamaCppServerManager - Specialized server manager for llama.cpp servers.

This extends the base ServerManager with llama.cpp-specific functionality.
"""

import os
from pathlib import Path
from typing import List, Optional

from models import Model, ModelProfile, UserConfig
from models.config_utils import (
    resolve_gpu_config,
    resolve_parameter_optimization_config,
)
from runner.server_manager.base import BaseServerManager


class LlamaCppServerManager(BaseServerManager):
    """Manages llama.cpp server process lifecycle."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        user_config: Optional[UserConfig] = None,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        # Resolve startup timeout from config - use longer timeout for large models
        poc = resolve_parameter_optimization_config(profile, user_config)
        startup_timeout = (
            poc.startup_timeout
            if poc and hasattr(poc, "startup_timeout")
            else 120  # Increased from 30 to 120 seconds
        )

        super().__init__(
            model=model,
            profile=profile,
            user_config=user_config,
            port=port,
            startup_timeout=startup_timeout,
        )
        self.is_embedding = is_embedding

    def get_api_endpoint(self, path: str) -> str:
        """Get the full URL for a specific API endpoint."""
        # For llama.cpp, most endpoints use /v1 prefix except health/metrics
        if path in ["/health", "/metrics"]:
            return f"{self.server_url}{path}"
        else:
            return f"{self.server_url}/v1{path}"

    def get_gguf_path(self) -> str:
        """Return resolved GGUF file path for model."""
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model

    def _build_server_args(self) -> List[str]:
        """Build command line arguments for llama.cpp server."""
        gguf_path = self.get_gguf_path()

        # Base command
        args = [
            "/llama.cpp/build/bin/llama-server",
            "--model",
            gguf_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
        ]

        # Add embedding-specific flags early
        if self.is_embedding:
            args.extend(
                [
                    "--threads",
                    str(os.cpu_count() or 4),
                    "--ctx-size",
                    "4096",  # Smaller context for embeddings
                    "--batch-size",
                    "1024",
                    "--embeddings",  # Enable embeddings mode
                    "--pooling",
                    "mean",  # Use mean pooling
                    "--no-webui",  # Disable web UI
                ]
            )

            # Add debug logging if enabled
            if os.getenv("LOG_LEVEL", "WARNING").lower() == "trace":
                args.extend(["--verbose"])

            self._logger.info(f"Embedding server args: {' '.join(args)}")
            return args

        # For non-embedding servers, build full configuration
        params = self.profile.parameters
        gcfg = resolve_gpu_config(self.profile, self.user_config)

        # Add standard server features with performance optimizations
        args.extend(
            [
                "--cont-batching",
                "--metrics",
                "--no-warmup",  # Skip warmup for faster startup
                "--cache-type-k", "f16",  # Use f16 for KV cache
                "--cache-type-v", "f16",  # Use f16 for KV cache
            ]
        )

        # Core performance parameters
        args.extend(["--threads", str(os.cpu_count() or 4)])
        args.extend(["-c", str(params.num_ctx or 90000)])
        args.extend(["--batch-size", str(params.batch_size or 256)])
        args.extend(["-ub", str(params.batch_size or 256)])

        # GPU configuration
        args.extend(
            [
                "--n-gpu-layers",
                str(gcfg.gpu_layers if gcfg.gpu_layers is not None else -1),
            ]
        )

        # Main GPU selection
        if gcfg.main_gpu is not None and gcfg.main_gpu >= 0:
            args.extend(["-mg", str(gcfg.main_gpu)])
        else:
            args.extend(["-mg", "1"])  # Default to GPU 1 for large models

        # Tensor split configuration
        if gcfg.tensor_split:
            tensor_split_str = ",".join(map(str, gcfg.tensor_split))
            args.extend(["-ts", tensor_split_str])

        # Split mode configuration
        if hasattr(gcfg, "split_mode") and gcfg.split_mode:
            args.extend(["-sm", str(gcfg.split_mode)])

        # MoE (Mixture of Experts) configuration
        if (
            hasattr(params, "n_cpu_moe")
            and params.n_cpu_moe is not None
            and params.n_cpu_moe > 0
        ):
            args.extend(["--n-cpu-moe", str(params.n_cpu_moe)])
        else:
            args.extend(["--n-cpu-moe", "5"])  # Default for MoE models

        # NUMA distribution
        args.extend(["--numa", "distribute"])

        # KV offload disable
        args.extend(["-nkvo"])

        # Multimodal support - critical for vision models
        mmproj_path = self._get_multimodal_projector_path(gguf_path)
        if mmproj_path and Path(mmproj_path).exists():
            args.extend(["--mmproj", mmproj_path])
            self._logger.info(f"Using multimodal projector: {mmproj_path}")
        elif "vl" in self.model.name.lower() or "vision" in self.model.name.lower():
            self._logger.warning(
                f"Vision model detected but no mmproj file found for {self.model.name}"
            )

        # Draft model support for speculative decoding
        if hasattr(self.profile, "draft_model") and self.profile.draft_model:
            args.extend(["--model-draft", str(self.profile.draft_model)])

        # Additional GPU optimizations
        if hasattr(gcfg, "offload_kqv") and not gcfg.offload_kqv:
            args.extend(["--no-kv-offload"])

        # Disable web UI for server mode
        args.extend(["--no-webui"])

        # Enable JSON schema support for tools (required for tool calling)
        args.extend(["--jinja"])

        # Add logging configuration
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "debug":
            args.extend(["--verbose"])

        self._logger.info(f"Server args: {' '.join(args)}")
        return args

    def _get_multimodal_projector_path(self, gguf_path: str) -> Optional[str]:
        """Find multimodal projector file for vision models."""
        # Check various possible locations for mmproj file
        if (
            hasattr(self.model.details, "clip_model_path")
            and self.model.details.clip_model_path
        ):
            return self.model.details.clip_model_path
        elif (
            hasattr(self.model.details, "mmproj_path")
            and self.model.details.mmproj_path
        ):
            return self.model.details.mmproj_path
        elif (
            hasattr(self.model.details, "parent_model")
            and self.model.details.parent_model
            and "qwen" in self.model.details.parent_model.lower()
        ):
            # For Qwen models, try to find mmproj in same directory as model
            model_dir = Path(gguf_path).parent
            possible_mmproj = model_dir / "mmproj.gguf"
            if possible_mmproj.exists():
                return str(possible_mmproj)
            else:
                # Try alternative naming patterns
                possible_mmproj = model_dir / "mmproj-model-f16.gguf"
                if possible_mmproj.exists():
                    return str(possible_mmproj)

        return None
