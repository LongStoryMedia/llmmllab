"""
Llama.cpp Argument Builder - Builds command-line arguments for llama.cpp servers.

Builds a config dict from model parameters, then serializes it directly
to a command-line argument list.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.model_parameters import ModelParameters
from models import Model, UserConfig
from config import LLAMA_SERVER_EXECUTABLE, LOG_LEVEL
from utils.logging import llmmllogger

logger = llmmllogger.bind(component="LlamaCppArgumentBuilder")


class LlamaCppArgumentBuilder:
    """Builds command-line arguments for llama.cpp servers.

    Usage::

        builder = LlamaCppArgumentBuilder(model, user_config, port)
        args = builder.build_args()   # ["/llama.cpp/.../llama-server", "--model", "/path/to.gguf", ...]
    """

    def __init__(
        self,
        model: Model,
        user_config: Optional[UserConfig] = None,
        port: Optional[int] = None,
        is_embedding: bool = False,
    ):
        self.model = model
        self.user_config = user_config
        self.port = port
        self.is_embedding = is_embedding

    def build_args(self) -> List[str]:
        """Build the complete argument list for the llama.cpp server process."""
        config = self._build_config()
        args = [LLAMA_SERVER_EXECUTABLE] + _config_to_args(config)
        logger.debug(f"Built args: {' '.join(args)}")
        return args

    def _build_config(self) -> Dict[str, Any]:
        """Build the full config dict for this model."""
        config: Dict[str, Any] = {
            "model": self._get_gguf_path(),
            "host": "127.0.0.1",
            "port": self.port,
        }

        if self.is_embedding:
            self._add_embedding_config(config)
        else:
            self._add_inference_config(config)

        return config

    # ── Embedding ────────────────────────────────────────────────────

    def _add_embedding_config(self, config: Dict[str, Any]) -> None:
        config.update(
            {
                "threads": os.cpu_count() or 4,
                "ctx_size": 4096,
                "batch_size": 1024,
                "embedding": True,
                "pooling": "mean",
                "no_webui": True,
            }
        )
        if LOG_LEVEL.lower() == "debug":
            config["verbose"] = True

    # ── Inference ────────────────────────────────────────────────────

    def _add_inference_config(self, config: Dict[str, Any]) -> None:
        params = self.model.parameters or ModelParameters()

        config.update(
            {
                "cont_batching": True,
                "metrics": True,
                "slots": True,
                "no_warmup": True,
                "flash_attn": "on",
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "threads": int(os.cpu_count() or 4),
                "ctx_size": params.num_ctx or 90000,
                "batch_size": params.batch_size or 2048,
                "ubatch_size": params.micro_batch_size or (params.batch_size or 2048),
                "reasoning": "on" if params.think else "off",
                "reasoning_budget": 16384 if params.think else 0,
                "ctx_checkpoints": 24,
                "timeout": 600,
                "context_shift": True,
                "mirostat": 1,
                "cache_ram": 0,
                "repeat_penalty": params.repeat_penalty or 1.1,
                "repeat_last_n": (
                    params.repeat_last_n if params.repeat_last_n is not None else 256
                ),
                "n_gpu_layers": (
                    params.n_gpu_layers if params.n_gpu_layers is not None else -1
                ),
                "main_gpu": params.main_gpu if params.main_gpu is not None else -1,
                "split_mode": params.split_mode or "layer",
                # "parallel": 1,
                "jinja": True,
                "no_webui": True,
            }
        )

        # Tensor split — only set if configured
        if params.tensor_split:
            config["tensor_split"] = params.tensor_split

        # KV cache placement:
        #   --no-kv-offload tells llama.cpp to keep KV on CPU.
        #   Without it, KV is offloaded to GPU alongside model layers.
        config["no_kv_offload"] = params.kv_on_cpu

        # Multimodal projector
        mmproj_path = self.model.details.clip_model_path
        if mmproj_path and Path(mmproj_path).exists():
            config["mmproj"] = mmproj_path
            logger.info(f"Using multimodal projector: {mmproj_path}")
        elif "vl" in self.model.name.lower() or "vision" in self.model.name.lower():
            logger.warning(
                f"Vision model detected but no mmproj file found for {self.model.name}"
            )

        # Draft model (speculative decoding)
        if hasattr(self.model, "draft_model") and self.model.draft_model:
            if mmproj_path and Path(mmproj_path).exists():
                logger.warning(
                    f"Draft models are not supported with multimodal models. "
                    f"Ignoring draft model for {self.model.name}"
                )
            else:
                from runner.utils.model_loader import ModelLoader

                ml = ModelLoader()
                dm = ml.get_model_by_id(self.model.draft_model)
                draft_gguf = dm.details.gguf_file if dm and dm.details else None
                if draft_gguf:
                    config["model_draft"] = str(draft_gguf)

        if LOG_LEVEL.lower() == "trace":
            config["verbose"] = True

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_gguf_path(self) -> str:
        details = getattr(self.model, "details", None)
        if details and hasattr(details, "gguf_file") and details.gguf_file:
            return details.gguf_file
        return self.model.model


def _config_to_args(config: Dict[str, Any]) -> List[str]:
    """Convert a {flag_name: value} dict to a flat command-line arg list.

    Keys use underscores (python style); they are converted to hyphens
    for the CLI.  Booleans emit a bare flag when True, nothing when False.
    None values are skipped.
    """
    args: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, (list, tuple)):
            if value:
                args.extend([flag, ",".join(map(str, value))])
        else:
            args.extend([flag, str(value)])
    return args
