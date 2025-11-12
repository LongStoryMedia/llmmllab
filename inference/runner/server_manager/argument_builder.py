"""
Argument Builder - Structured flag management using argparse for server configurations.

This module provides a clean, maintainable way to build server command-line arguments
using argparse's infrastructure without actually parsing command line arguments.
"""

import argparse
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from models import Model, ModelProfile, UserConfig
from models.config_utils import resolve_gpu_config, resolve_parameter_optimization_config
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="ArgumentBuilder")


class BaseArgumentBuilder(ABC):
    """Abstract base class for building server arguments using argparse."""

    def __init__(
        self,
        model: Model,
        profile: ModelProfile,
        user_config: Optional[UserConfig] = None,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        self.model = model
        self.profile = profile
        self.user_config = user_config
        self.port = port
        self.is_embedding = is_embedding
        self._parser = None
        self._args = None
        self._setup_parser()

    @abstractmethod
    def _setup_parser(self) -> None:
        """Setup the argument parser with server-specific flags."""

    @abstractmethod
    def _get_executable_path(self) -> str:
        """Return the path to the server executable."""

    def _create_parser(self, description: str) -> argparse.ArgumentParser:
        """Create a new argument parser with common settings."""
        return argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            add_help=False,  # We're not parsing user input
        )

    def _add_common_args(self) -> None:
        """Add common arguments shared across server types."""
        if not self._parser:
            return

        # Basic server configuration
        self._parser.add_argument("--host", default="127.0.0.1")
        self._parser.add_argument("--port", type=int, default=self.port)

        # Performance
        self._parser.add_argument("--threads", type=int, default=os.cpu_count() or 4)

        # Logging
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "trace":
            self._parser.add_argument("--verbose", action="store_true", default=True)

    def build_args(self) -> List[str]:
        """Build the complete argument list for the server."""
        if not self._parser or not self._args:
            raise RuntimeError("Parser not properly initialized")

        # Convert namespace to argument list
        args = [self._get_executable_path()]

        # Convert argument namespace to command line format
        for key, value in vars(self._args).items():
            if value is None:
                continue

            # Convert argument name to flag format
            flag = f"--{key.replace('_', '-')}"

            if isinstance(value, bool):
                if value:  # Only add flag if True
                    args.append(flag)
            elif isinstance(value, (list, tuple)):
                if value:  # Only add if not empty
                    args.extend([flag, ",".join(map(str, value))])
            else:
                args.extend([flag, str(value)])

        logger.debug(f"Built args: {' '.join(args)}")
        return args

    def get_args_dict(self) -> Dict[str, Any]:
        """Get arguments as a dictionary for inspection."""
        if not self._args:
            return {}
        return vars(self._args).copy()


class LlamaCppArgumentBuilder(BaseArgumentBuilder):
    """Argument builder for llama.cpp servers."""

    def _get_executable_path(self) -> str:
        """Return the path to llama.cpp server executable."""
        return "/llama.cpp/build/bin/llama-server"

    def _setup_parser(self) -> None:
        """Setup llama.cpp specific argument parser."""
        self._parser = self._create_parser("llama.cpp server arguments")

        # Model and basic configuration
        self._parser.add_argument("--model", required=True)
        self._add_common_args()

        # Context and batching
        self._parser.add_argument("-c", "--ctx-size", type=int, dest="ctx_size")
        self._parser.add_argument("--batch-size", type=int, dest="batch_size")
        self._parser.add_argument("-ub", "--ubatch-size", type=int, dest="ubatch_size")

        # GPU configuration
        self._parser.add_argument("--n-gpu-layers", type=int, dest="n_gpu_layers")
        self._parser.add_argument("-mg", "--main-gpu", type=int, dest="main_gpu")
        self._parser.add_argument("-ts", "--tensor-split", dest="tensor_split")
        self._parser.add_argument("-sm", "--split-mode", type=int, dest="split_mode")

        # Performance optimizations
        self._parser.add_argument("--cont-batching", action="store_true")
        self._parser.add_argument("--metrics", action="store_true")
        self._parser.add_argument("--no-warmup", action="store_true", dest="no_warmup")
        self._parser.add_argument("--cache-type-k", dest="cache_type_k")
        self._parser.add_argument("--cache-type-v", dest="cache_type_v")

        # MoE configuration
        self._parser.add_argument("--n-cpu-moe", type=int, dest="n_cpu_moe")

        # NUMA and memory
        self._parser.add_argument("--numa")
        self._parser.add_argument("-nkvo", "--no-kv-offload", action="store_true", dest="no_kv_offload")

        # Multimodal
        self._parser.add_argument("--mmproj")
        self._parser.add_argument("--model-draft", dest="model_draft")

        # Embedding specific
        self._parser.add_argument("--embeddings", action="store_true")
        self._parser.add_argument("--pooling")

        # UI and tools
        self._parser.add_argument("--no-webui", action="store_true", dest="no_webui")
        self._parser.add_argument("--jinja", action="store_true")

        # Build the arguments based on configuration
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
        config.update({
            "threads": os.cpu_count() or 4,
            "ctx_size": 4096,  # Smaller context for embeddings
            "batch_size": 1024,
            "embeddings": True,
            "pooling": "mean",
            "no_webui": True,
        })

        # Add debug logging if enabled
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "trace":
            config["verbose"] = True

    def _build_inference_config(self, config: Dict[str, Any]) -> None:
        """Build configuration for inference servers."""
        params = self.profile.parameters
        gcfg = resolve_gpu_config(self.profile, self.user_config)

        # Standard server features with performance optimizations
        config.update({
            "cont_batching": True,
            "metrics": True,
            "no_warmup": True,  # Skip warmup for faster startup
            "cache_type_k": "f16",  # Use f16 for KV cache
            "cache_type_v": "f16",  # Use f16 for KV cache
        })

        # Core performance parameters
        config.update({
            "threads": os.cpu_count() or 4,
            "ctx_size": params.num_ctx or 90000,
            "batch_size": params.batch_size or 256,
            "ubatch_size": params.batch_size or 256,
        })

        # GPU configuration
        config["n_gpu_layers"] = gcfg.gpu_layers if gcfg.gpu_layers is not None else -1

        # Main GPU selection
        if gcfg.main_gpu is not None and gcfg.main_gpu >= 0:
            config["main_gpu"] = gcfg.main_gpu
        else:
            config["main_gpu"] = 1  # Default to GPU 1 for large models

        # Tensor split configuration
        if gcfg.tensor_split:
            config["tensor_split"] = ",".join(map(str, gcfg.tensor_split))

        # Split mode configuration
        if hasattr(gcfg, "split_mode") and gcfg.split_mode:
            # Convert string split modes to integer values
            split_mode_mapping = {
                "layer": 1,  # LLAMA_SPLIT_MODE_LAYER
                "row": 2,    # LLAMA_SPLIT_MODE_ROW
            }
            if isinstance(gcfg.split_mode, str):
                split_mode = split_mode_mapping.get(gcfg.split_mode.lower(), 1)
            else:
                split_mode = gcfg.split_mode
            config["split_mode"] = split_mode

        # MoE (Mixture of Experts) configuration
        if (
            hasattr(params, "n_cpu_moe")
            and params.n_cpu_moe is not None
            and params.n_cpu_moe > 0
        ):
            config["n_cpu_moe"] = params.n_cpu_moe
        else:
            config["n_cpu_moe"] = 5  # Default for MoE models

        # NUMA distribution
        config["numa"] = "distribute"

        # KV offload disable
        config["no_kv_offload"] = True

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
        if hasattr(gcfg, "offload_kqv") and not gcfg.offload_kqv:
            config["no_kv_offload"] = True

        # Enable JSON schema support for tools (required for tool calling)
        config.update({
            "jinja": True,
            "no_webui": True,
        })

        # Add logging configuration
        if os.getenv("LOG_LEVEL", "WARNING").lower() == "trace":
            config["verbose"] = True

    def _get_gguf_path(self) -> str:
        """Return resolved GGUF file path for model."""
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model


# Factory function for creating argument builders
def create_argument_builder(
    server_type: str,
    model: Model,
    profile: ModelProfile,
    user_config: Optional[UserConfig] = None,
    port: Optional[int] = None,
    is_embedding: bool = False,
) -> BaseArgumentBuilder:
    """Create an argument builder for the specified server type."""
    builders = {
        "llamacpp": LlamaCppArgumentBuilder,
    }

    if server_type not in builders:
        raise ValueError(f"Unknown server type: {server_type}. Available: {list(builders.keys())}")

    return builders[server_type](
        model=model,
        profile=profile,
        user_config=user_config,
        port=port,
        is_embedding=is_embedding,
    )