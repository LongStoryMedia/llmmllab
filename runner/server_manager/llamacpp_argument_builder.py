"""
Llama.cpp Argument Builder - Specific implementation for llama.cpp servers.

This module provides the concrete implementation for building arguments
for llama.cpp servers with dynamic flag discovery and model-specific
configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict

from runner.models.config_utils import resolve_gpu_config
from runner.utils.logging import llmmllogger

from .base_argument_builder import BaseArgumentBuilder
from .dynamic_flag_parser import DynamicFlagParser

logger = llmmllogger.bind(component="LlamaCppArgumentBuilder")


class LlamaCppArgumentBuilder(BaseArgumentBuilder):
    """Argument builder for llama.cpp servers with dynamic flag discovery."""

    def _get_executable_path(self) -> str:
        """Return the path to llama.cpp server executable."""
        return "/llama.cpp/build/bin/llama-server"

    def _setup_parser(self) -> None:
        """Setup llama.cpp specific argument parser with dynamically discovered flags."""
        self._parser = self._create_parser("llama.cpp server arguments")

        # Add common arguments first
        self._add_common_args()

        # Use dynamic flag parser to discover and add all available flags
        dynamic_parser = DynamicFlagParser(self._get_executable_path())
        dynamic_parser.build_parser(self._parser)

        # Build the arguments based on configuration if model is available
        if hasattr(self, "model") and self.model:
            self._build_configuration()

    def _build_configuration(self) -> None:
        """Build the argument configuration based on model and profile."""
        config = {}

        # Get GGUF path
        gguf_path = self._get_gguf_path()
        config["model"] = gguf_path

        # Basic server config
        config["host"] = "127.0.0.1"
        config["port"] = self.port

        if self.is_embedding:
            self._build_embedding_config(config)
        else:
            self._build_inference_config(config)

        # Parse the configuration into arguments
        # We create a fake argument list and parse it
        fake_args = []
        for key, value in config.items():
            if value is None:
                continue

            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    fake_args.append(flag)
            elif isinstance(value, (list, tuple)):
                if value:
                    fake_args.extend([flag, ",".join(map(str, value))])
            else:
                fake_args.extend([flag, str(value)])

        self._args = self._parser.parse_args(fake_args)

    def _build_embedding_config(self, config: Dict[str, Any]) -> None:
        """Build configuration for embedding servers."""
        config.update(
            {
                "threads": os.cpu_count() or 4,
                "ctx_size": 4096,  # Smaller context for embeddings
                "batch_size": 1024,
                "embedding": True,  # Singular 'embedding' to match llama.cpp flag
                "pooling": "mean",
                "no_webui": True,  # Disable web UI for embedding servers
            }
        )

        # Add debug logging if enabled
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "debug":
            config["verbose"] = True

    def _build_inference_config(self, config: Dict[str, Any]) -> None:
        """Build configuration for inference servers."""
        params = self.profile.parameters
        gcfg = resolve_gpu_config(self.profile, self.user_config)

        # Standard server features with performance optimizations
        config.update(
            {
                # "cont_batching": True,
                # "metrics": True,
                "no_warmup": True,  # Skip warmup for faster startup
                "flash_attn": "on",  # Flash attention for faster prompt processing
                "cache_type_k": "f16",  # Use f16 for KV cache
                "cache_type_v": "f16",  # Use f16 for KV cache
            }
        )

        # Parallelism: use 1 slot to avoid GGML_ASSERT failures with
        # GQA models on multi-GPU row-split and to prevent KV cache
        # thrashing under SWA/hybrid attention models.
        config["parallel"] = 1

        # Core performance parameters
        config.update(
            {
                "threads": (int(os.cpu_count() or 5) - 1),
                "ctx_size": params.num_ctx or 90000,
                "batch_size": params.batch_size or 2048,
                "ubatch_size": params.micro_batch_size or (params.batch_size or 2048),
                "reasoning_budget": (-1 if self.profile.parameters.think else 0),
            }
        )

        # GPU configuration
        config["n_gpu_layers"] = (
            params.n_gpu_layers
            if params.n_gpu_layers is not None
            else (gcfg.gpu_layers if gcfg.gpu_layers is not None else -1)
        )

        # Main GPU selection
        if gcfg.main_gpu is not None and gcfg.main_gpu >= 0:
            config["main_gpu"] = gcfg.main_gpu

        # Tensor split configuration
        if gcfg.tensor_split:
            config["tensor_split"] = ",".join(map(str, gcfg.tensor_split))

        # Split mode configuration
        if hasattr(gcfg, "split_mode") and gcfg.split_mode:
            # Pass string split modes directly to llama.cpp
            if isinstance(gcfg.split_mode, str):
                config["split_mode"] = gcfg.split_mode.lower()
            else:
                # Convert legacy integer values to strings
                split_mode_mapping = {
                    1: "layer",  # LLAMA_SPLIT_MODE_LAYER
                    2: "row",  # LLAMA_SPLIT_MODE_ROW
                }
                config["split_mode"] = split_mode_mapping.get(gcfg.split_mode, "layer")

        # MoE (Mixture of Experts) configuration
        config["n_cpu_moe"] = params.n_cpu_moe

        # NUMA distribution
        # config["numa"] = "distribute"

        # KV cache configuration: offload = move to CPU, no_offload = keep on GPU
        # kv_on_cpu=True means user wants KV on CPU, so enable offloading
        # kv_on_cpu=False means user wants KV on GPU, so disable offloading (keep on GPU)
        config["no_kv_offload"] = not params.kv_on_cpu

        # config["flash_attn"] = (
        #     "on"
        #     if params.flash_attention
        #     else "off" if not params.flash_attention else "auto"
        # )

        # Multimodal support - critical for vision models
        mmproj_path = self.model.details.clip_model_path
        if mmproj_path and Path(mmproj_path).exists():
            config["mmproj"] = mmproj_path
            logger.info(f"Using multimodal projector: {mmproj_path}")
        elif "vl" in self.model.name.lower() or "vision" in self.model.name.lower():
            logger.warning(
                f"Vision model detected but no mmproj file found for {self.model.name}"
            )

        # Draft model support for speculative decoding
        if hasattr(self.profile, "draft_model") and self.profile.draft_model:
            if mmproj_path and Path(mmproj_path).exists():
                logger.warning(
                    f"Draft models are not supported with multimodal models. Ignoring draft model for {self.model.name}"
                )
            else:
                from runner.utils.model_loader import ModelLoader

                ml = ModelLoader()
                dm = ml.get_model_by_id(self.profile.draft_model)
                draft_gguf = dm.details.gguf_file if dm and dm.details else None
                if draft_gguf:
                    config["model_draft"] = str(draft_gguf)

        # Additional GPU optimizations
        # if hasattr(gcfg, "offload_kqv") and not gcfg.offload_kqv:
        #     config["no_kv_offload"] = True

        # Enable JSON schema support for tools (required for tool calling)
        config.update(
            {
                "jinja": True,
                "no_webui": True,  # Explicitly disable web UI features for server mode
            }
        )
        #
        # Add logging configuration
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "trace":
            config["verbose"] = True

    def _get_gguf_path(self) -> str:
        """Return resolved GGUF file path for model."""
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model
