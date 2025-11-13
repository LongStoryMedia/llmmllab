"""
Argument Builder - Structured flag management using argparse for server configurations.

This module provides a clean, maintainable way to build server command-line arguments
using argparse's infrastructure without actually parsing command line arguments.
"""

import argparse
import os
import re
import subprocess
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


class DynamicFlagParser:
    """Dynamically parse llama.cpp server flags from help output."""
    
    def __init__(self, executable_path: str):
        self.executable_path = executable_path
        self.parsed_flags = None
        
    def get_help_output(self) -> str:
        """Get the help output from the llama-server executable."""
        try:
            result = subprocess.run(
                [self.executable_path, "--help"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
            else:
                logger.warning(f"Help command failed with code {result.returncode}: {result.stderr}")
                return ""
        except Exception as e:
            logger.error(f"Failed to get help output from {self.executable_path}: {e}")
            return ""
    
    def parse_flags(self) -> List[Dict[str, Any]]:
        """Parse flag definitions from help output."""
        if self.parsed_flags is not None:
            return self.parsed_flags
            
        help_output = self.get_help_output()
        if not help_output:
            logger.warning("No help output available, falling back to static flags")
            return []
            
        flags = []
        
        # Parse line by line looking for flag definitions
        lines = help_output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines, section headers, and non-flag lines
            # Use regex to match actual flags: start with 1-2 dashes + alphabetic character
            if not line or line.startswith('-----') or not re.match(r'^-{1,2}[a-zA-Z]', line):
                continue
            
            # Parse flag line - format is typically:
            # "-t, --threads N                      number of CPU threads..."
            # "--help                               print usage and exit"
            # "-ngl, --gpu-layers, --n-gpu-layers N  max. number of layers..."
            
            # Split to separate flags from description - look for pattern where 
            # flags end and description begins (description doesn't start with -)
            parts = None
            
            # Try to split on large whitespace gaps, but ensure the description part 
            # doesn't start with a flag (-)
            for potential_split in re.finditer(r'\s{2,}', line):
                start = potential_split.start()
                potential_desc = line[potential_split.end():].strip()
                
                # Description shouldn't start with a flag marker
                if potential_desc and not potential_desc.startswith('-'):
                    flag_spec = line[:start].strip()
                    description = potential_desc
                    parts = [flag_spec, description]
                    break
            
            # If no description found on same line, treat entire line as flag spec
            if not parts or len(parts) < 2:
                # This line contains only flag specification, no description
                flag_spec = line.strip()
                description = ""  # Empty description
                parts = [flag_spec, description]
            
            flag_spec = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else ""
            
            # Parse the flag specification with improved logic
            short_flags = []
            long_flags = []
            value_type = None
            
            # Split on comma first, then parse each part
            flag_parts = [f.strip() for f in flag_spec.split(',') if f.strip()]
            
            for part in flag_parts:
                part = part.strip()
                
                # Split the part to separate flag name from value type
                tokens = part.split()
                if not tokens:
                    continue
                    
                flag_name = tokens[0]
                
                # Check for value type after flag name
                if len(tokens) > 1:
                    potential_value_type = tokens[1]
                    if potential_value_type.upper() in ['N', 'TYPE', 'SEED', 'FNAME', 'HOST', 'PORT', 'PATH', 'PREFIX', 'KEY', 'TOKEN', 'STRING', 'SCHEMA', 'FILE', 'URL', 'SCALE', 'INDEX', 'SIMILARITY', 'FORMAT', 'SEQUENCE', 'SAMPLERS', 'PROMPT', 'GRAMMAR', 'BIAS', 'JINJA_TEMPLATE', 'JINJA_TEMPLATE_FILE', 'M', 'P', '<0|1>', '<0...100>', 'LO-HI']:
                        value_type = potential_value_type
                
                # Categorize flag
                if flag_name.startswith('--'):
                    long_flags.append(flag_name)
                elif flag_name.startswith('-') and len(flag_name) > 1:
                    short_flags.append(flag_name)
            
            # Determine if this flag takes a value
            takes_value = value_type is not None
            
            # Enhance value type detection from description
            if not takes_value:
                desc_lower = description.lower()
                if any(pattern in desc_lower for pattern in [
                    'number of', 'size of', 'path to', 'url', 'file', 'directory', 
                    'timeout', 'port', 'host', 'value', 'factor', 'probability',
                    'temperature', 'scale', 'rate', 'threshold'
                ]):
                    takes_value = True
            
            # Determine argument type
            arg_type = self._infer_argument_type(description, takes_value, value_type)
            
            if long_flags or short_flags:
                flag_info = {
                    'short_flags': short_flags,
                    'long_flags': long_flags, 
                    'type': arg_type['type'],
                    'action': arg_type['action'],
                    'help': description,
                    'takes_value': takes_value,
                    'value_type': value_type
                }
                flags.append(flag_info)
        
        self.parsed_flags = flags
        logger.info(f"Parsed {len(flags)} flags from llama.cpp help output")
        return flags
    
    def _infer_argument_type(self, description: str, takes_value: bool, value_type: Optional[str] = None) -> Dict[str, Any]:
        """Infer argument type from description and value requirement."""
        desc_lower = description.lower()
        
        # Boolean flags (no value)
        if not takes_value or any(word in desc_lower for word in ['enable', 'disable', 'default: false', 'default: true']):
            return {'type': None, 'action': 'store_true'}
        
        # String flags based on value type indicators (highest priority)
        if value_type and value_type.upper() in [
            'FNAME', 'FILE', 'PATH', 'URL', 'HOST', 'TOKEN', 'KEY', 'STRING', 
            'SCHEMA', 'SAMPLERS', 'PROMPT', 'GRAMMAR', 'BIAS', 'SEQUENCE',
            'JINJA_TEMPLATE', 'JINJA_TEMPLATE_FILE', 'FORMAT', 'PREFIX',
            'TYPE', 'SEED', 'SCALE', 'INDEX', 'SIMILARITY', 'M', 'LO-HI',
            '<0|1>', '<0...100>', '<DEV1'  # Special patterns
        ]:
            return {'type': str, 'action': 'store'}
        
        # Integer flags based on value type indicator
        if value_type == 'N' or any(word in desc_lower for word in ['number', 'size', 'count', 'threads', 'layers', 'index', 'port', 'timeout']):
            return {'type': int, 'action': 'store'}
            
        # Float flags - be more specific to avoid false positives
        if value_type == 'P' or any(word in desc_lower for word in ['temperature', 'probability', 'factor', 'threshold', 'penalty', 'learning rate', 'ratio']):
            return {'type': float, 'action': 'store'}
        
        # String flags (default for value-taking flags)
        return {'type': str, 'action': 'store'}
    
    def build_parser(self, base_parser: argparse.ArgumentParser) -> None:
        """Add dynamically discovered flags to the argument parser."""
        flags = self.parse_flags()
        
        added_count = 0
        for flag_info in flags:
            try:
                # Combine short and long flags
                flag_names = flag_info['short_flags'] + flag_info['long_flags']
                if not flag_names:
                    continue
                
                # Build argparse arguments
                kwargs = {
                    'help': flag_info['help']
                }
                
                if flag_info['action'] == 'store_true':
                    kwargs['action'] = 'store_true'
                else:
                    if flag_info['type']:
                        kwargs['type'] = flag_info['type']
                
                # Add the argument
                base_parser.add_argument(*flag_names, **kwargs)
                added_count += 1
                
            except argparse.ArgumentError as e:
                # Skip conflicting arguments (probably already defined)
                logger.debug(f"Skipping conflicting flag {flag_info.get('long_flags', flag_info.get('short_flags'))}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to add flag {flag_info}: {e}")
                continue
        
        logger.info(f"Added {added_count} dynamic flags to argument parser")


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