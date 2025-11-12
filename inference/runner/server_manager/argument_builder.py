"""
Argument Builder - Structured flag management using argparse for server configurations.

This module provides a clean, maintainable way to build server command-line arguments
using argparse's infrastructure without actually parsing command line arguments.
"""

import argparse
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from models import Model, ModelProfile, UserConfig
from models.config_utils import resolve_gpu_config
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
        self._parser.add_argument("--host", default="127.0.0.1", help="IP address to listen on")
        self._parser.add_argument("--port", type=int, default=8080, help="Port to listen on")

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
        """Setup llama.cpp specific argument parser with all available flags."""
        self._parser = self._create_parser("llama.cpp server arguments")

        # Help and version
        self._parser.add_argument("--help", "--usage", action="store_true", help="Print usage and exit")
        self._parser.add_argument("--version", action="store_true", help="Show version and build info")
        self._parser.add_argument("--completion-bash", action="store_true", help="Print source-able bash completion script")
        self._parser.add_argument("--verbose-prompt", action="store_true", dest="verbose_prompt", help="Print a verbose prompt before generation")

        # Model and basic configuration
        self._parser.add_argument("-m", "--model", required=True, help="Model path")
        self._parser.add_argument("-mu", "--model-url", dest="model_url", help="Model download URL")
        self._parser.add_argument("-hf", "-hfr", "--hf-repo", dest="hf_repo", help="Hugging Face model repository")
        self._parser.add_argument("-hfd", "-hfrd", "--hf-repo-draft", dest="hf_repo_draft", help="Hugging Face draft model repository")
        self._parser.add_argument("-hff", "--hf-file", dest="hf_file", help="Hugging Face model file")
        self._parser.add_argument("-hfv", "-hfrv", "--hf-repo-v", dest="hf_repo_v", help="Hugging Face vocoder model repository")
        self._parser.add_argument("-hffv", "--hf-file-v", dest="hf_file_v", help="Hugging Face vocoder model file")
        self._parser.add_argument("-hft", "--hf-token", dest="hf_token", help="Hugging Face access token")
        self._parser.add_argument("-a", "--alias", help="Set alias for model name")

        self._add_common_args()

        # Threading and CPU configuration
        self._parser.add_argument("-t", "--threads", type=int, help="Number of threads to use during generation")
        self._parser.add_argument("-tb", "--threads-batch", type=int, dest="threads_batch", help="Number of threads for batch processing")
        self._parser.add_argument("-C", "--cpu-mask", dest="cpu_mask", help="CPU affinity mask")
        self._parser.add_argument("-Cr", "--cpu-range", dest="cpu_range", help="Range of CPUs for affinity")
        self._parser.add_argument("--cpu-strict", type=int, dest="cpu_strict", help="Use strict CPU placement")
        self._parser.add_argument("--prio", type=int, help="Set process/thread priority")
        self._parser.add_argument("--poll", type=int, help="Use polling level to wait for work")
        self._parser.add_argument("-Cb", "--cpu-mask-batch", dest="cpu_mask_batch", help="CPU affinity mask for batch")
        self._parser.add_argument("-Crb", "--cpu-range-batch", dest="cpu_range_batch", help="CPU range for batch")
        self._parser.add_argument("--cpu-strict-batch", type=int, dest="cpu_strict_batch", help="Use strict CPU placement for batch")
        self._parser.add_argument("--prio-batch", type=int, dest="prio_batch", help="Set batch process/thread priority")
        self._parser.add_argument("--poll-batch", type=int, dest="poll_batch", help="Use polling for batch")

        # Context and batching
        self._parser.add_argument("-c", "--ctx-size", type=int, dest="ctx_size", help="Size of the prompt context")
        self._parser.add_argument("-n", "--predict", "--n-predict", type=int, dest="n_predict", help="Number of tokens to predict")
        self._parser.add_argument("-b", "--batch-size", type=int, dest="batch_size", help="Logical maximum batch size")
        self._parser.add_argument("-ub", "--ubatch-size", type=int, dest="ubatch_size", help="Physical maximum batch size")
        self._parser.add_argument("--keep", type=int, help="Number of tokens to keep from initial prompt")
        self._parser.add_argument("--swa-full", action="store_true", dest="swa_full", help="Use full-size SWA cache")
        self._parser.add_argument("--kv-unified", "-kvu", action="store_true", dest="kv_unified", help="Use single unified KV buffer")
        self._parser.add_argument("-fa", "--flash-attn", action="store_true", dest="flash_attn", help="Enable Flash Attention")
        self._parser.add_argument("--no-perf", action="store_true", dest="no_perf", help="Disable internal performance timings")

        # Text processing
        self._parser.add_argument("-e", "--escape", action="store_true", help="Process escape sequences")
        self._parser.add_argument("--no-escape", action="store_true", dest="no_escape", help="Do not process escape sequences")

        # RoPE configuration
        self._parser.add_argument("--rope-scaling", help="RoPE frequency scaling method")
        self._parser.add_argument("--rope-scale", type=float, dest="rope_scale", help="RoPE context scaling factor")
        self._parser.add_argument("--rope-freq-base", type=float, dest="rope_freq_base", help="RoPE base frequency")
        self._parser.add_argument("--rope-freq-scale", type=float, dest="rope_freq_scale", help="RoPE frequency scaling factor")
        self._parser.add_argument("--yarn-orig-ctx", type=int, dest="yarn_orig_ctx", help="YaRN original context size")
        self._parser.add_argument("--yarn-ext-factor", type=float, dest="yarn_ext_factor", help="YaRN extrapolation mix factor")
        self._parser.add_argument("--yarn-attn-factor", type=float, dest="yarn_attn_factor", help="YaRN scale sqrt(t) or attention magnitude")
        self._parser.add_argument("--yarn-beta-slow", type=float, dest="yarn_beta_slow", help="YaRN high correction dim or alpha")
        self._parser.add_argument("--yarn-beta-fast", type=float, dest="yarn_beta_fast", help="YaRN low correction dim or beta")

        # KV cache configuration  
        self._parser.add_argument("-nkvo", "--no-kv-offload", action="store_true", dest="no_kv_offload", help="Disable KV offload")
        self._parser.add_argument("-nr", "--no-repack", action="store_true", dest="no_repack", help="Disable weight repacking")
        self._parser.add_argument("-ctk", "--cache-type-k", dest="cache_type_k", help="KV cache data type for K")
        self._parser.add_argument("-ctv", "--cache-type-v", dest="cache_type_v", help="KV cache data type for V")
        self._parser.add_argument("-dt", "--defrag-thold", type=int, dest="defrag_thold", help="KV cache defragmentation threshold")
        self._parser.add_argument("-np", "--parallel", type=int, help="Number of parallel sequences to decode")

        # Memory configuration
        self._parser.add_argument("--mlock", action="store_true", help="Force system to keep model in RAM")
        self._parser.add_argument("--no-mmap", action="store_true", dest="no_mmap", help="Do not memory-map model")
        self._parser.add_argument("--numa", help="NUMA optimization type")

        # GPU configuration
        self._parser.add_argument("-dev", "--device", help="Comma-separated list of devices for offloading")
        self._parser.add_argument("--list-devices", action="store_true", dest="list_devices", help="Print list of available devices")
        self._parser.add_argument("--override-tensor", "-ot", dest="override_tensor", help="Override tensor buffer type")
        self._parser.add_argument("--cpu-moe", "-cmoe", action="store_true", dest="cpu_moe", help="Keep MoE weights in CPU")
        self._parser.add_argument("--n-cpu-moe", "-ncmoe", type=int, dest="n_cpu_moe", help="Keep MoE weights of first N layers in CPU")
        self._parser.add_argument("-ngl", "--gpu-layers", "--n-gpu-layers", type=int, dest="n_gpu_layers", help="Number of layers to store in VRAM")
        self._parser.add_argument("-sm", "--split-mode", dest="split_mode", help="How to split the model across multiple GPUs")
        self._parser.add_argument("-ts", "--tensor-split", dest="tensor_split", help="Fraction of model to offload to each GPU")
        self._parser.add_argument("-mg", "--main-gpu", type=int, dest="main_gpu", help="The GPU to use for the model")
        self._parser.add_argument("--check-tensors", action="store_true", dest="check_tensors", help="Check model tensor data for invalid values")
        self._parser.add_argument("--override-kv", dest="override_kv", help="Override model metadata by key")
        self._parser.add_argument("--no-op-offload", action="store_true", dest="no_op_offload", help="Disable offloading host tensor operations")

        # LoRA and control vectors
        self._parser.add_argument("--lora", help="Path to LoRA adapter")
        self._parser.add_argument("--lora-scaled", dest="lora_scaled", help="Path to LoRA adapter with scaling")
        self._parser.add_argument("--lora-base", dest="lora_base", help="Path to base model for LoRA")
        self._parser.add_argument("--control-vector", dest="control_vector", help="Add a control vector")
        self._parser.add_argument("--control-vector-scaled", dest="control_vector_scaled", help="Add a control vector with scaling")
        self._parser.add_argument("--control-vector-layer-range", dest="control_vector_layer_range", help="Layer range to apply control vectors")

        # Multimodal
        self._parser.add_argument("--mmproj", help="Path to multimodal projector file")
        self._parser.add_argument("--mmproj-url", dest="mmproj_url", help="URL to multimodal projector file")
        self._parser.add_argument("--no-mmproj", action="store_true", dest="no_mmproj", help="Disable multimodal projector")
        self._parser.add_argument("--no-mmproj-offload", action="store_true", dest="no_mmproj_offload", help="Do not offload multimodal projector to GPU")

        # Draft model configuration
        self._parser.add_argument("--override-tensor-draft", "-otd", dest="override_tensor_draft", help="Override tensor buffer type for draft model")
        self._parser.add_argument("--cpu-moe-draft", "-cmoed", action="store_true", dest="cpu_moe_draft", help="Keep MoE weights in CPU for draft model")
        self._parser.add_argument("--n-cpu-moe-draft", "-ncmoed", type=int, dest="n_cpu_moe_draft", help="Keep MoE weights of first N layers in CPU for draft")
        self._parser.add_argument("-md", "--model-draft", dest="model_draft", help="Draft model for speculative decoding")
        self._parser.add_argument("--spec-replace", dest="spec_replace", help="Translate string in TARGET into DRAFT")
        self._parser.add_argument("-mv", "--model-vocoder", dest="model_vocoder", help="Vocoder model for audio generation")
        self._parser.add_argument("--tts-use-guide-tokens", action="store_true", dest="tts_use_guide_tokens", help="Use guide tokens for TTS")

        # Logging configuration
        self._parser.add_argument("--log-disable", action="store_true", dest="log_disable", help="Disable logging")
        self._parser.add_argument("--log-file", dest="log_file", help="Log to file")
        self._parser.add_argument("--log-colors", action="store_true", dest="log_colors", help="Enable colored logging")
        self._parser.add_argument("-v", "--verbose", "--log-verbose", action="store_true", dest="verbose", help="Set verbosity to infinity")
        self._parser.add_argument("--offline", action="store_true", help="Offline mode")
        self._parser.add_argument("-lv", "--verbosity", "--log-verbosity", type=int, dest="log_verbosity", help="Set verbosity threshold")
        self._parser.add_argument("--log-prefix", action="store_true", dest="log_prefix", help="Enable prefix in log messages")
        self._parser.add_argument("--log-timestamps", action="store_true", dest="log_timestamps", help="Enable timestamps in log messages")

        # Draft model KV cache
        self._parser.add_argument("-ctkd", "--cache-type-k-draft", dest="cache_type_k_draft", help="KV cache data type for K for draft model")
        self._parser.add_argument("-ctvd", "--cache-type-v-draft", dest="cache_type_v_draft", help="KV cache data type for V for draft model")

        # Sampling parameters
        self._parser.add_argument("--samplers", help="Samplers for generation")
        self._parser.add_argument("-s", "--seed", type=int, help="RNG seed")
        self._parser.add_argument("--sampling-seq", "--sampler-seq", dest="sampling_seq", help="Simplified sequence for samplers")
        self._parser.add_argument("--ignore-eos", action="store_true", dest="ignore_eos", help="Ignore end of stream token")
        self._parser.add_argument("--temp", type=float, help="Temperature")
        self._parser.add_argument("--top-k", type=int, dest="top_k", help="Top-k sampling")
        self._parser.add_argument("--top-p", type=float, dest="top_p", help="Top-p sampling")
        self._parser.add_argument("--min-p", type=float, dest="min_p", help="Min-p sampling")
        self._parser.add_argument("--top-nsigma", type=float, dest="top_nsigma", help="Top-n-sigma sampling")
        self._parser.add_argument("--xtc-probability", type=float, dest="xtc_probability", help="XTC probability")
        self._parser.add_argument("--xtc-threshold", type=float, dest="xtc_threshold", help="XTC threshold")
        self._parser.add_argument("--typical", type=float, help="Locally typical sampling")
        self._parser.add_argument("--repeat-last-n", type=int, dest="repeat_last_n", help="Last n tokens to consider for penalize")
        self._parser.add_argument("--repeat-penalty", type=float, dest="repeat_penalty", help="Penalize repeat sequence of tokens")
        self._parser.add_argument("--presence-penalty", type=float, dest="presence_penalty", help="Repeat alpha presence penalty")
        self._parser.add_argument("--frequency-penalty", type=float, dest="frequency_penalty", help="Repeat alpha frequency penalty")
        self._parser.add_argument("--dry-multiplier", type=float, dest="dry_multiplier", help="Set DRY sampling multiplier")
        self._parser.add_argument("--dry-base", type=float, dest="dry_base", help="Set DRY sampling base value")
        self._parser.add_argument("--dry-allowed-length", type=int, dest="dry_allowed_length", help="Set allowed length for DRY sampling")
        self._parser.add_argument("--dry-penalty-last-n", type=int, dest="dry_penalty_last_n", help="Set DRY penalty for last n tokens")
        self._parser.add_argument("--dry-sequence-breaker", dest="dry_sequence_breaker", help="Add sequence breaker for DRY sampling")
        self._parser.add_argument("--dynatemp-range", type=float, dest="dynatemp_range", help="Dynamic temperature range")
        self._parser.add_argument("--dynatemp-exp", type=float, dest="dynatemp_exp", help="Dynamic temperature exponent")
        self._parser.add_argument("--mirostat", type=int, help="Use Mirostat sampling")
        self._parser.add_argument("--mirostat-lr", type=float, dest="mirostat_lr", help="Mirostat learning rate")
        self._parser.add_argument("--mirostat-ent", type=float, dest="mirostat_ent", help="Mirostat target entropy")
        self._parser.add_argument("-l", "--logit-bias", dest="logit_bias", help="Modify likelihood of token appearing")
        self._parser.add_argument("--grammar", help="BNF-like grammar to constrain generations")
        self._parser.add_argument("--grammar-file", dest="grammar_file", help="File to read grammar from")
        self._parser.add_argument("-j", "--json-schema", dest="json_schema", help="JSON schema to constrain generations")
        self._parser.add_argument("-jf", "--json-schema-file", dest="json_schema_file", help="File containing JSON schema")

        # Example-specific parameters
        self._parser.add_argument("--swa-checkpoints", type=int, dest="swa_checkpoints", help="Max number of SWA checkpoints per slot")
        self._parser.add_argument("--no-context-shift", action="store_true", dest="no_context_shift", help="Disable context shift")
        self._parser.add_argument("--context-shift", action="store_true", dest="context_shift", help="Enable context shift")
        self._parser.add_argument("-r", "--reverse-prompt", dest="reverse_prompt", help="Halt generation at PROMPT")
        self._parser.add_argument("-sp", "--special", action="store_true", help="Special tokens output enabled")
        self._parser.add_argument("--no-warmup", action="store_true", dest="no_warmup", help="Skip warming up the model")
        self._parser.add_argument("--spm-infill", action="store_true", dest="spm_infill", help="Use Suffix/Prefix/Middle pattern for infill")
        self._parser.add_argument("--pooling", help="Pooling type for embeddings")
        self._parser.add_argument("-cb", "--cont-batching", action="store_true", dest="cont_batching", help="Enable continuous batching")
        self._parser.add_argument("-nocb", "--no-cont-batching", action="store_true", dest="no_cont_batching", help="Disable continuous batching")

        # Server configuration
        self._parser.add_argument("--path", help="Path to serve static files from")
        self._parser.add_argument("--api-prefix", dest="api_prefix", help="Prefix path the server serves from")
        self._parser.add_argument("--no-webui", action="store_true", dest="no_webui", help="Disable the Web UI")
        self._parser.add_argument("--embedding", "--embeddings", action="store_true", dest="embeddings", help="Restrict to embedding use case")
        self._parser.add_argument("--reranking", "--rerank", action="store_true", dest="reranking", help="Enable reranking endpoint")
        self._parser.add_argument("--api-key", dest="api_key", help="API key for authentication")
        self._parser.add_argument("--api-key-file", dest="api_key_file", help="Path to file containing API keys")
        self._parser.add_argument("--ssl-key-file", dest="ssl_key_file", help="Path to SSL private key file")
        self._parser.add_argument("--ssl-cert-file", dest="ssl_cert_file", help="Path to SSL certificate file")
        self._parser.add_argument("--chat-template-kwargs", dest="chat_template_kwargs", help="Additional params for JSON template parser")
        self._parser.add_argument("-to", "--timeout", type=int, help="Server read/write timeout in seconds")
        self._parser.add_argument("--threads-http", type=int, dest="threads_http", help="Number of threads for HTTP requests")
        self._parser.add_argument("--cache-reuse", type=int, dest="cache_reuse", help="Min chunk size to attempt reusing from cache")
        self._parser.add_argument("--metrics", action="store_true", help="Enable prometheus compatible metrics endpoint")
        self._parser.add_argument("--props", action="store_true", help="Enable changing global properties via POST /props")
        self._parser.add_argument("--slots", action="store_true", help="Enable slots monitoring endpoint")
        self._parser.add_argument("--no-slots", action="store_true", dest="no_slots", help="Disable slots monitoring endpoint")
        self._parser.add_argument("--slot-save-path", dest="slot_save_path", help="Path to save slot kv cache")
        self._parser.add_argument("--jinja", action="store_true", help="Use jinja template for chat")
        self._parser.add_argument("--reasoning-format", dest="reasoning_format", help="Controls whether thought tags are allowed")
        self._parser.add_argument("--reasoning-budget", type=int, dest="reasoning_budget", help="Controls amount of thinking allowed")
        self._parser.add_argument("--chat-template", dest="chat_template", help="Set custom jinja chat template")
        self._parser.add_argument("--chat-template-file", dest="chat_template_file", help="Set custom jinja chat template file")
        self._parser.add_argument("--no-prefill-assistant", action="store_true", dest="no_prefill_assistant", help="Disable prefilling assistant response")
        self._parser.add_argument("-sps", "--slot-prompt-similarity", type=float, dest="slot_prompt_similarity", help="Prompt similarity for slot reuse")
        self._parser.add_argument("--lora-init-without-apply", action="store_true", dest="lora_init_without_apply", help="Load LoRA adapters without applying")

        # Draft/speculative decoding
        self._parser.add_argument("-td", "--threads-draft", type=int, dest="threads_draft", help="Number of threads for draft model")
        self._parser.add_argument("-tbd", "--threads-batch-draft", type=int, dest="threads_batch_draft", help="Number of threads for batch processing draft")
        self._parser.add_argument("--draft-max", "--draft", "--draft-n", type=int, dest="draft_max", help="Number of tokens to draft for speculative decoding")
        self._parser.add_argument("--draft-min", "--draft-n-min", type=int, dest="draft_min", help="Minimum number of draft tokens")
        self._parser.add_argument("--draft-p-min", type=float, dest="draft_p_min", help="Minimum speculative decoding probability")
        self._parser.add_argument("-cd", "--ctx-size-draft", type=int, dest="ctx_size_draft", help="Size of prompt context for draft model")
        self._parser.add_argument("-devd", "--device-draft", dest="device_draft", help="Devices for offloading draft model")
        self._parser.add_argument("-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft", type=int, dest="n_gpu_layers_draft", help="Number of layers in VRAM for draft")

        # Default model variants (convenience flags)
        self._parser.add_argument("--embd-bge-small-en-default", action="store_true", dest="embd_bge_small_en_default", help="Use default bge-small-en-v1.5 model")
        self._parser.add_argument("--embd-e5-small-en-default", action="store_true", dest="embd_e5_small_en_default", help="Use default e5-small-v2 model")
        self._parser.add_argument("--embd-gte-small-default", action="store_true", dest="embd_gte_small_default", help="Use default gte-small model")
        self._parser.add_argument("--fim-qwen-1.5b-default", action="store_true", dest="fim_qwen_1_5b_default", help="Use default Qwen 2.5 Coder 1.5B")
        self._parser.add_argument("--fim-qwen-3b-default", action="store_true", dest="fim_qwen_3b_default", help="Use default Qwen 2.5 Coder 3B")
        self._parser.add_argument("--fim-qwen-7b-default", action="store_true", dest="fim_qwen_7b_default", help="Use default Qwen 2.5 Coder 7B")
        self._parser.add_argument("--fim-qwen-7b-spec", action="store_true", dest="fim_qwen_7b_spec", help="Use Qwen 2.5 Coder 7B + 0.5B draft")
        self._parser.add_argument("--fim-qwen-14b-spec", action="store_true", dest="fim_qwen_14b_spec", help="Use Qwen 2.5 Coder 14B + 0.5B draft")
        self._parser.add_argument("--fim-qwen-30b-default", action="store_true", dest="fim_qwen_30b_default", help="Use default Qwen 3 Coder 30B A3B Instruct")

        # Build the arguments based on configuration if model is available
        if hasattr(self, 'model') and self.model:
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
            # Pass string split modes directly to llama.cpp
            if isinstance(gcfg.split_mode, str):
                config["split_mode"] = gcfg.split_mode.lower()
            else:
                # Convert legacy integer values to strings
                split_mode_mapping = {
                    1: "layer",  # LLAMA_SPLIT_MODE_LAYER
                    2: "row",    # LLAMA_SPLIT_MODE_ROW
                }
                config["split_mode"] = split_mode_mapping.get(gcfg.split_mode, "layer")

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