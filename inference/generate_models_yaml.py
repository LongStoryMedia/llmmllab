#!/usr/bin/env python3
"""
Generate model definitions in YAML format by scanning GGUF files under /models.

Usage (inside container):
    python3 /scripts/generate_models_yaml.py --models-dir /models --output /models/models.yaml

Or from host (using kubectl):
    kubectl exec -it <pod-name> -- python3 /scripts/generate_models_yaml.py --models-dir /models --output /models/models.yaml
"""

import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from functools import partial
from typing import Any

import gguf


def get_gguf_metadata(gguf_path: str) -> dict[str, Any]:
    """Extract metadata from a GGUF file using gguf module directly."""
    reader = gguf.GGUFReader(gguf_path)
    metadata = {}
    for key, value in reader.fields.items():
        val = value.contents()
        if isinstance(val, list) and len(val) == 1:
            val = val[0]
        metadata[key] = val
    return metadata


def calculate_digest(file_path: str) -> str:
    """Calculate SHA256 digest of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(partial(f.read, 65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_model_task(metadata: dict[str, Any]) -> str:
    """Detect the model task type from metadata."""
    arch = metadata.get("general.architecture", "").lower()
    description = metadata.get("general.description", "").lower()
    general_type = metadata.get("general.type", "").lower()

    # Text-to-text (LLM)
    llm_archs = {
        "llama",
        "llava",
        "phi3",
        "gemma",
        "mistral",
        "qwen",
        "deepseek",
        "stablelm",
        "opt",
        "bloom",
        "falcon",
        "gpt2",
        "gptj",
        "gptneox",
        "mpt",
        "codegen",
        "refact",
        "nemotron",
        "granite",
        "gemma2",
        "command-r",
        "dbrx",
        "exaone",
        "grit",
        "kunyu",
        "minicpm",
        "olmo",
        "openelm",
        "orion",
        "persimmon",
        "phi3small",
        "phi4",
        "qwen2",
        "qwen3",
        "starcoder",
        "jamba",
        "mixtral",
        "yi",
        "chatglm",
        "internlm",
        "baichuan",
        "qwen2.5",
        "llama3",
        "llama3.1",
        "llama3.2",
    }

    # Text-to-image (SDXL, SD3, etc.)
    img2img_archs = {
        "stable-diffusion",
        "sdxl",
        "sd3",
        "flux",
        "pixtral",
        "kandinsky",
        "controlnet",
        "inpaint",
        "upscaler",
    }

    # Image-to-text (vision models)
    vision_archs = {
        "llava",
        "idefics",
        "paligemma",
        "instructblip",
        "vision-encoder",
        "clip",
    }

    # Embeddings
    if "embedding" in description or "embeddings" in description:
        return "TextToEmbeddings"

    # Check general.type for mmproj (CLIP vision projector)
    if general_type == "mmproj":
        return "ImageToText"

    # Check architecture
    if any(a in arch for a in vision_archs):
        return "ImageToText"
    if any(a in arch for a in img2img_archs):
        return "TextToImage"
    if any(a in arch for a in llm_archs):
        return "TextToText"

    # Default
    return "TextToText"


def detect_model_family(metadata: dict[str, Any]) -> str:
    """Detect the model family from metadata."""
    arch = metadata.get("general.architecture", "").lower()

    # Map architecture to family - handles both base and variant architectures
    arch_family_map = {
        "llama": "llama",
        "llava": "llava",
        "phi3": "phi3",
        "phi4": "phi4",
        "gemma": "gemma",
        "mistral": "mistral",
        "qwen": "qwen",
        "qwen2": "qwen2",
        "qwen3": "qwen3",
        "qwen3next": "qwen",  # Qwen3-Coder-Next maps to qwen family
        "qwen35moe": "qwen",  # Qwen3.5 MoE maps to qwen family
        "deepseek": "deepseek",
        "stablelm": "stablelm",
        "stable-diffusion": "stable-diffusion",
        "sdxl": "sdxl",
        "sd3": "sd3",
        "flux": "flux",
        "opt": "opt",
        "bloom": "bloom",
        "falcon": "falcon",
        "gpt2": "gpt2",
        "gptj": "gptj",
        "gptneox": "gptneox",
        "mpt": "mpt",
        "codegen": "codegen",
        "refact": "refact",
        "nemotron": "nemotron",
        "granite": "granite",
        "gemma2": "gemma2",
        "command-r": "command-r",
        "dbrx": "dbrx",
        "exaone": "exaone",
        "grit": "grit",
        "kunyu": "kunyu",
        "minicpm": "minicpm",
        "olmo": "olmo",
        "openelm": "openelm",
        "orion": "orion",
        "persimmon": "persimmon",
        "phi3small": "phi3small",
        "starcoder": "starcoder",
        "jamba": "jamba",
        "mixtral": "mixtral",
        "yi": "yi",
        "chatglm": "chatglm",
        "internlm": "internlm",
        "baichuan": "baichuan",
        "nomic-bert-moe": "nomic-bert",
        "clip": "clip",
        "qwen35": "qwen",  # Qwen3.5 maps to qwen family
    }

    # Check general.architecture first
    if arch in arch_family_map:
        return arch_family_map[arch]

    # Try to extract from other keys
    for key in metadata.keys():
        key_lower = key.lower()
        if "architecture" in key_lower:
            for family in arch_family_map.values():
                if family in key_lower:
                    return family

    return "unknown"


def calculate_parameter_size_from_tensors(file_path: str) -> str:
    """Calculate parameter size from GGUF tensor data."""
    try:
        reader = gguf.GGUFReader(file_path)
        total_params = 0
        for tensor in reader.tensors:
            params = 1
            for d in tensor.shape:
                params *= d
            total_params += params
        # Convert to human-readable format
        if total_params >= 1e12:
            return f"{total_params / 1e12:.1f}T"
        elif total_params >= 1e9:
            return f"{total_params / 1e9:.1f}B"
        elif total_params >= 1e6:
            return f"{total_params / 1e6:.1f}M"
        elif total_params >= 1e3:
            return f"{total_params / 1e3:.1f}K"
        return f"{int(total_params)}"
    except Exception:
        return "unknown"


def create_model_definition(
    file_path: str, metadata: dict[str, Any], mmproj_map: dict[str, str]
) -> dict[str, Any]:
    """Create a model definition dict from GGUF metadata."""
    file_size = os.path.getsize(file_path)
    digest = calculate_digest(file_path)
    modified_at = datetime.now(timezone.utc).isoformat()

    # Extract GGUF metadata values
    arch = metadata.get(
        "general.architecture", metadata.get("llama.architecture", "")
    ).lower()
    name = metadata.get("general.name", os.path.basename(file_path))
    description = metadata.get("general.description", "")
    quantization = metadata.get("general.quantization_level", "")
    basename = metadata.get(
        "general.basename", os.path.basename(file_path).replace(".gguf", "")
    )

    # Extract quantization level from file name if not in metadata
    if not quantization:
        filename = os.path.basename(file_path)
        # Pattern: match any Q or IQ pattern at end of filename (handles q8_0, iq4_xs, q6_k_s, q8_k_xl, etc.)
        quant_match = re.search(
            r"^(ud-)?([a-z]*[QI][a-z0-9_]+)\.gguf$", filename, re.IGNORECASE
        )
        if quant_match:
            quantization = quant_match.group(2).upper()

    # Determine task
    task = detect_model_task(metadata)

    # Determine family and families
    family = detect_model_family(metadata)
    families = [family] if family != "unknown" else []

    # Get parameter size - use tensor-based calculation which is most accurate
    parameter_size = calculate_parameter_size_from_tensors(file_path)

    # Get context length - try architecture-specific keys first
    # Use 'or' chain to handle None/0 values properly
    context_length = (
        metadata.get("llama.context_length")
        or metadata.get("phi3.context_length")
        or metadata.get("qwen3next.context_length")
        or metadata.get("qwen35moe.context_length")
        or metadata.get("clip.vision.context_length")
        or metadata.get("general.context_length")
        or 2048
    )

    # Determine precision from quantization
    precision = "unknown"
    dtype = "unknown"
    if quantization:
        quant_lower = str(quantization).lower()
        if "bf16" in quant_lower:
            precision = "bf16"
            dtype = "bfloat16"
        elif "fp16" in quant_lower or "f16" in quant_lower:
            precision = "fp16"
            dtype = "float16"
        elif "fp32" in quant_lower or "f32" in quant_lower:
            precision = "fp32"
            dtype = "float32"
        elif "q4" in quant_lower:
            precision = "int4"
            dtype = "int4"
        elif "q8" in quant_lower:
            precision = "int8"
            dtype = "int8"
        elif "q6" in quant_lower:
            precision = "int6"
            dtype = "int6"
        elif "q5" in quant_lower:
            precision = "int5"
            dtype = "int5"
        elif "q3" in quant_lower:
            precision = "int3"
            dtype = "int3"

    # Determine specialization based on model type
    specialization = "Text"
    if (
        "mmproj" in file_path.lower()
        or "clip" in metadata.get("general.description", "").lower()
        or metadata.get("general.type") == "mmproj"
    ):
        specialization = "Vision"
    elif (
        "embed" in file_path.lower()
        or "embedding" in metadata.get("general.description", "").lower()
    ):
        specialization = "Embedding"

    # Determine task from general.tags
    general_tags = metadata.get("general.tags", [])
    if isinstance(general_tags, list):
        tags_lower = [str(t).lower() for t in general_tags]
        if "text-to-image" in tags_lower:
            task = "TextToImage"
        elif "image-to-text" in tags_lower:
            task = "ImageToText"
        elif "image-to-image" in tags_lower:
            task = "ImageToImage"
        elif "text-to-video" in tags_lower:
            task = "TextToVideo"
        elif "video-to-text" in tags_lower:
            task = "VideoToText"
        elif "speech-to-text" in tags_lower:
            task = "SpeechToText"
        elif "text-to-speech" in tags_lower:
            task = "TextToSpeech"

    # Add clip_model_path for mmproj models
    # Look up mmproj path from mmproj_map if available
    clip_model_path = ""
    if "mmproj" in file_path.lower() or metadata.get("general.type") == "mmproj":
        clip_model_path = file_path
    else:
        # Check if there's an mmproj in the same directory
        parent_dir = os.path.dirname(file_path)
        clip_model_path = mmproj_map.get(parent_dir, "")

    # Build details
    details: dict[str, Any] = {
        "parent_model": metadata.get("general.parent_model", ""),
        "format": "gguf",
        "gguf_file": file_path,
        "clip_model_path": clip_model_path,
        "family": family,
        "families": families,
        "parameter_size": parameter_size,
        "quantization_level": str(quantization) if quantization else "unknown",
        "precision": precision,
        "dtype": dtype,
        "specialization": specialization,
        "description": description,
        "weight": 1.0,
        "size": file_size,
        "original_ctx": (
            int(context_length) if isinstance(context_length, (int, float)) else 2048
        ),
    }

    # Add optional metadata if available - try architecture-specific keys first
    # n_layers (block_count)
    n_layers = (
        metadata.get("llama.block_count")
        or metadata.get("phi3.block_count")
        or metadata.get("qwen3next.block_count")
        or metadata.get("qwen35moe.block_count")
        or metadata.get("clip.vision.block_count")
        or metadata.get("nomic-bert-moe.block_count")
    )
    if n_layers:
        details["n_layers"] = int(n_layers)

    # hidden_size (embedding_length)
    n_embd = (
        metadata.get("llama.embedding_length")
        or metadata.get("phi3.embedding_length")
        or metadata.get("qwen3next.embedding_length")
        or metadata.get("qwen35moe.embedding_length")
        or metadata.get("clip.vision.embedding_length")
        or metadata.get("nomic-bert-moe.embedding_length")
    )
    if n_embd:
        details["hidden_size"] = int(n_embd)

    # n_heads (attention.head_count)
    n_heads = (
        metadata.get("llama.attention.head_count")
        or metadata.get("phi3.attention.head_count")
        or metadata.get("qwen3next.attention.head_count")
        or metadata.get("qwen35moe.attention.head_count")
        or metadata.get("clip.vision.attention.head_count")
        or metadata.get("nomic-bert-moe.attention.head_count")
    )
    if n_heads:
        details["n_heads"] = int(n_heads)

    # n_kv_heads (attention.head_count_kv)
    n_kv_heads = (
        metadata.get("llama.attention.head_count_kv")
        or metadata.get("phi3.attention.head_count_kv")
        or metadata.get("qwen3next.attention.head_count_kv")
        or metadata.get("qwen35moe.attention.head_count_kv")
        or metadata.get("clip.vision.attention.head_count_kv")
        or metadata.get("nomic-bert-moe.attention.head_count_kv")
    )
    if n_kv_heads:
        details["n_kv_heads"] = int(n_kv_heads)

    # Add dtype if determined
    if dtype != "unknown":
        details["dtype"] = dtype

    # Determine pipeline based on model type
    pipeline = "llama"
    if "sdxl" in arch or "stable-diffusion" in arch:
        pipeline = "sdxl"
    elif "sd3" in arch:
        pipeline = "sd3"
    elif "flux" in arch:
        pipeline = "flux"
    elif "llava" in arch:
        pipeline = "llava"
    elif "clip" in arch:
        pipeline = "clip"

    # Build LoRA weights list (empty for now)
    lora_weights = []

    # Build provider
    provider = "llama_cpp"

    # Build model definition
    model_def = {
        "id": basename.replace("-", "_").replace(".", "_"),
        "name": str(name) if name else os.path.basename(file_path),
        "model": basename,
        "task": task,
        "modified_at": modified_at,
        "digest": digest,
        "size": file_size,
        "details": details,
        "pipeline": pipeline,
        "lora_weights": lora_weights,
        "provider": provider,
    }

    return model_def


def find_gguf_files(root_dir: str) -> tuple[list[str], dict[str, str]]:
    """
    Recursively find all GGUF files under root_dir.
    Returns a tuple of (model_files, mmproj_map) where:
    - model_files: list of non-mmproj GGUF files
    - mmproj_map: dict mapping model file paths to their mmproj file paths
    """
    gguf_files = []
    mmproj_map = {}
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".gguf"):
                file_path = os.path.join(root, file)
                if "mmproj" in file.lower() or "clip" in file.lower():
                    # Store mmproj file path keyed by parent directory
                    parent_dir = os.path.dirname(file_path)
                    # Map by parent directory to find associated model
                    mmproj_map[parent_dir] = file_path
                else:
                    gguf_files.append(file_path)
    return sorted(gguf_files), mmproj_map


def models_to_yaml(models: list[dict[str, Any]]) -> str:
    """Convert list of model definitions to YAML format."""
    lines = []
    lines.append("# Generated model definitions")
    lines.append(
        f"# Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    lines.append("models:")

    for model in models:
        lines.append("")
        lines.append("  - id: " + escape_yaml_value(model.get("id", "")))
        lines.append("    name: " + escape_yaml_value(model.get("name", "")))
        lines.append("    model: " + escape_yaml_value(model.get("model", "")))
        lines.append("    task: " + model.get("task", ""))
        lines.append(f'    modified_at: "{model.get("modified_at", "")}"')
        lines.append("    digest: " + model.get("digest", ""))
        lines.append("    size: " + str(model.get("size", 0)))
        lines.append("    details:")
        details = model.get("details", {})
        lines.append("      format: " + escape_yaml_value(details.get("format", "")))
        lines.append(
            "      gguf_file: " + escape_yaml_value(details.get("gguf_file", ""))
        )
        lines.append("      family: " + escape_yaml_value(details.get("family", "")))
        lines.append("      families:")
        for fam in details.get("families", []):
            lines.append("        - " + escape_yaml_value(fam))
        lines.append(
            "      parameter_size: "
            + escape_yaml_value(details.get("parameter_size", ""))
        )

        if "quantization_level" in details:
            lines.append(
                "      quantization_level: "
                + escape_yaml_value(str(details.get("quantization_level", "")))
            )
        if "precision" in details:
            lines.append(
                "      precision: " + escape_yaml_value(details.get("precision", ""))
            )
        if "dtype" in details:
            lines.append("      dtype: " + escape_yaml_value(details.get("dtype", "")))
        if "specialization" in details:
            lines.append(
                "      specialization: "
                + escape_yaml_value(details.get("specialization", ""))
            )

        lines.append("      size: " + str(details.get("size", 0)))
        lines.append("      original_ctx: " + str(details.get("original_ctx", 0)))
        if "clip_model_path" in details and details["clip_model_path"]:
            lines.append(
                "      clip_model_path: "
                + escape_yaml_value(details.get("clip_model_path", ""))
            )

        if "n_layers" in details:
            lines.append("      n_layers: " + str(details.get("n_layers", 0)))
        if "hidden_size" in details:
            lines.append("      hidden_size: " + str(details.get("hidden_size", 0)))
        if "n_heads" in details:
            lines.append("      n_heads: " + str(details.get("n_heads", 0)))
        if "n_kv_heads" in details:
            lines.append("      n_kv_heads: " + str(details.get("n_kv_heads", 0)))
        if "description" in details and details["description"]:
            lines.append(
                "      description: "
                + escape_yaml_value(str(details.get("description", "")))
            )

        lines.append("    pipeline: " + escape_yaml_value(model.get("pipeline", "")))
        lines.append("    lora_weights: []")
        lines.append("    provider: " + model.get("provider", ""))

    return "\n".join(lines) + "\n"


def escape_yaml_value(value: str) -> str:
    """Escape a value for YAML output."""
    if not value:
        return '""'
    # Check if value needs quoting
    if any(
        c in value
        for c in [
            ":",
            "#",
            "[",
            "]",
            "{",
            "}",
            ",",
            "%",
            "&",
            "*",
            "?",
            "|",
            "-",
            "<",
            ">",
            "=",
            "!",
            "@",
            "`",
            "'",
            '"',
            "\n",
            "\t",
        ]
    ):
        # Escape quotes and wrap in quotes
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def main():
    """main entry point"""
    parser = argparse.ArgumentParser(
        description="Generate model definitions from GGUF files"
    )
    parser.add_argument(
        "--models-dir",
        default="/models",
        help="Directory to scan for GGUF files (default: /models)",
    )
    parser.add_argument(
        "--output",
        help="Output YAML file path (default: stdout if not specified)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Find all GGUF files
    print(f"Scanning {args.models_dir} for GGUF files...")
    gguf_files, mmproj_map = find_gguf_files(args.models_dir)
    print(f"Found {len(gguf_files)} GGUF file(s)")

    if not gguf_files:
        print("No GGUF files found. Exiting.")
        sys.exit(0)

    # Process each file
    models = []
    for i, gguf_path in enumerate(gguf_files, 1):
        if args.verbose:
            print(f"[{i}/{len(gguf_files)}] Processing: {gguf_path}")

        # Get metadata
        metadata = get_gguf_metadata(gguf_path)

        # Create model definition
        model_def = create_model_definition(gguf_path, metadata, mmproj_map)
        models.append(model_def)

        if args.verbose:
            print(
                f"  -> Model: {model_def['name']}, Task: {model_def['task']}, Pipeline: {model_def['pipeline']}"
            )

    # Sort models by name
    models.sort(key=lambda m: m["name"])

    # Convert to YAML
    yaml_output = models_to_yaml(models)

    # Write output
    if args.output:
        output_path = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(yaml_output)
        print(f"Wrote {len(models)} model definition(s) to {output_path}")
    else:
        # Output to stdout
        print(yaml_output, end="")


if __name__ == "__main__":
    main()
