from typing import Optional, Dict, TypedDict

from models import Model, ModelDetails, OptimalParameters


import re
from typing import Optional, List, Literal, Annotated, Dict, Tuple
from pydantic import BaseModel, Field

# (Paste the updated ModelDetails, Model, and OptimalParameters classes here)


class MemoryBreakdown(TypedDict):
    """Type definition for memory breakdown return values."""

    model_weights_gpu_gb: float
    clip_model_gb: float
    kv_cache_gb: float
    activation_gb: float
    overhead_gb: float
    total_gpu_gb: float
    cpu_memory_gb: float
    gpu_layers_loaded: int
    total_layers: int
    quantization_bits: int
    model_size_b: float
    model_size_gb: float
    hidden_size: int
    n_heads: int
    n_kv_heads: int


class Resizer:
    """
    Calculate GPU memory requirements for LLM inference, optimized for llama.cpp.

    Formulas are based on common llama.cpp memory components:
    1.  **Model Weights:** The portion of the model file (GGUF) loaded onto the GPU.
    2.  **KV Cache:** Memory to store keys and values for past tokens. This scales
        with context size, layers on GPU, and GQA factor.
    3.  **Activation/Compute Buffer:** "Scratch" memory used during prompt processing
        (ingestion), which is the peak memory usage. This scales with context
        size and micro-batch size.
    4.  **Overhead:** Fixed cost for the CUDA context, kernels, and fragmentation.
    """

    @staticmethod
    def _parse_parameter_size(param_size: str) -> float:
        """
        Parse parameter size string (e.g., '7.2B', '13B', '8x7B', '70B') to float.
        """
        param_size = param_size.upper().strip()

        # --- IMPROVEMENT: Handle MoE models like '8x7B' ---
        moe_match = re.match(r"(\d+)X(\d+\.?\d*)B", param_size)
        if moe_match:
            # For MoE, we care about the expert size for architecture,
            # but the total size for file size. Let's return the
            # *total* parameters for estimation if needed.
            # However, the 'model.size' (file size) is the most accurate
            # measure for weights, so this parser is mainly for heuristics.
            # We'll return the expert size (e.g., 7.0 for 8x7B) as it's
            # more indicative of the *architecture* (layers, hidden size).
            # Let's refine: 8x7B model has 7B expert params.
            # 8x22B model has 22B expert params.
            return float(moe_match.group(2))

        if param_size.endswith("B"):
            return float(param_size[:-1])
        elif param_size.endswith("M"):
            return float(param_size[:-1]) / 1000.0
        elif param_size.endswith("K"):
            return float(param_size[:-1]) / 1_000_000.0
        else:
            try:
                return float(param_size)
            except ValueError as e:
                raise ValueError(f"Unable to parse parameter size: {param_size}") from e

    @staticmethod
    def _parse_quantization_bits(
        quantization_level: Optional[str], precision: Optional[str]
    ) -> int:
        """
        Extract bits per weight from quantization level or precision.
        """
        # Try quantization_level first
        if quantization_level:
            q_upper = quantization_level.upper()

            # --- IMPROVEMENT: Handle F16/F32 directly ---
            if "F32" in q_upper or "FP32" in q_upper:
                return 32
            if "F16" in q_upper or "FP16" in q_upper:
                return 16

            if "Q2" in q_upper or "IQ2" in q_upper:
                return 2
            elif "Q3" in q_upper or "IQ3" in q_upper:
                return 3
            elif "Q4" in q_upper or "IQ4" in q_upper:
                return 4
            elif "Q5" in q_upper:
                return 5
            elif "Q6" in q_upper:
                return 6
            elif "Q8" in q_upper:
                return 8

        # Try precision
        if precision:
            p_lower = precision.lower()
            if "fp32" in p_lower:
                return 32
            elif "fp16" in p_lower or "bf16" in p_lower:
                return 16
            elif "int8" in p_lower:
                return 8
            elif "int4" in p_lower:
                return 4
            elif "int2" in p_lower:
                return 2
            elif "int1" in p_lower:
                return 1

        # Default to 4-bit if nothing found (common for GGUF models)
        return 4

    @staticmethod
    def _get_model_architecture_details(
        model_size_b: float, family: str
    ) -> Tuple[int, int, int, int]:
        """
        --- NEW HEURISTIC FUNCTION ---
        Estimate architecture details if not provided.
        This is a *heuristic* and is much less accurate than providing
        the values directly in ModelDetails.

        Returns:
            Tuple[total_layers, hidden_size, n_heads, n_kv_heads]
        """
        family_lower = family.lower()

        # Default (fallback)
        total_layers, hidden_size, n_heads, n_kv_heads = 32, 4096, 32, 32

        if "qwen" in family_lower:
            if model_size_b <= 0.5:  # Qwen2-0.5B
                total_layers, hidden_size, n_heads, n_kv_heads = 24, 896, 14, 2
            elif model_size_b <= 1.5:  # Qwen2-1.5B
                total_layers, hidden_size, n_heads, n_kv_heads = 28, 1536, 12, 2
            elif model_size_b <= 7:  # Qwen2-7B
                total_layers, hidden_size, n_heads, n_kv_heads = 28, 3584, 28, 4
            elif model_size_b <= 14:  # Qwen2-57B-A14B (MoE expert size)
                # Heuristic: Assume 14B expert is similar to a 7B model
                total_layers, hidden_size, n_heads, n_kv_heads = 28, 3584, 28, 4
            elif model_size_b <= 72:  # Qwen2-72B
                total_layers, hidden_size, n_heads, n_kv_heads = 80, 8192, 64, 8
            else:  # Fallback for larger Qwen
                total_layers, hidden_size, n_heads, n_kv_heads = 80, 8192, 64, 8

            return total_layers, hidden_size, n_heads, n_kv_heads

        if "phi" in family_lower:
            if model_size_b <= 4:  # Phi-3 Mini (3.8B)
                total_layers, hidden_size, n_heads, n_kv_heads = 32, 3072, 32, 32
            else:  # Phi-3 Small (7B)
                total_layers, hidden_size, n_heads, n_kv_heads = 32, 4096, 32, 8

        elif "mistral" in family_lower or "mixtral" in family_lower:
            # Mistral 7B / Mixtral 8x7B
            total_layers, hidden_size, n_heads, n_kv_heads = 32, 4096, 32, 8

        elif "llama" in family_lower or "gemma" in family_lower:
            if model_size_b <= 2:  # Gemma 2B
                total_layers, hidden_size, n_heads, n_kv_heads = (
                    22,
                    2560,
                    8,
                    8,
                )  # Heuristic
            elif model_size_b <= 7:  # Llama 2 7B / Gemma 7B
                total_layers, hidden_size, n_heads, n_kv_heads = 32, 4096, 32, 32
            elif model_size_b <= 9:  # Llama 3 8B / Gemma 2 9B
                total_layers, hidden_size, n_heads, n_kv_heads = 32, 4096, 32, 8
            elif model_size_b <= 13:  # Llama 2 13B
                total_layers, hidden_size, n_heads, n_kv_heads = 40, 5120, 40, 40
            elif model_size_b <= 34:  # Llama 1 34B
                total_layers, hidden_size, n_heads, n_kv_heads = (
                    60,
                    6656,
                    52,
                    52,
                )  # Heuristic
            elif model_size_b <= 70:  # Llama 2/3 70B
                total_layers, hidden_size, n_heads, n_kv_heads = 80, 8192, 64, 8
            else:
                total_layers, hidden_size, n_heads, n_kv_heads = 80, 8192, 64, 8

        # Apply generic scaling if family not recognized
        else:
            if model_size_b <= 3:
                total_layers, hidden_size, n_heads, n_kv_heads = 26, 2560, 32, 32
            elif model_size_b <= 7:
                total_layers, hidden_size, n_heads, n_kv_heads = 32, 4096, 32, 32
            elif model_size_b <= 13:
                total_layers, hidden_size, n_heads, n_kv_heads = 40, 5120, 40, 40
            elif model_size_b <= 30:
                total_layers, hidden_size, n_heads, n_kv_heads = 60, 6144, 48, 48
            elif model_size_b <= 70:
                total_layers, hidden_size, n_heads, n_kv_heads = (
                    80,
                    8192,
                    64,
                    8,
                )  # Assume GQA
            else:
                total_layers, hidden_size, n_heads, n_kv_heads = (
                    80,
                    10240,
                    80,
                    8,
                )  # Assume GQA

        return total_layers, hidden_size, n_heads, n_kv_heads

    def calculate_memory_breakdown(
        self, optimal_params: OptimalParameters, model: Model
    ) -> MemoryBreakdown:
        """
        Calculate detailed GPU memory requirements breakdown.
        """
        # --- 1. EXTRACT PARAMETERS ---

        model_size_b = self._parse_parameter_size(model.details.parameter_size)
        bits_per_weight = self._parse_quantization_bits(
            model.details.quantization_level, model.details.precision
        )

        # Get actual model file size in GB
        model_size_gb = model.details.size / (1024**3)

        # Extract optimal parameters
        n_ctx = optimal_params.n_ctx
        n_batch = optimal_params.n_batch
        n_ubatch = optimal_params.n_ubatch
        n_gpu_layers = optimal_params.n_gpu_layers

        # --- 2. GET MODEL ARCHITECTURE ---

        # (This section is unchanged - it gets total_layers, hidden_size, etc.)
        if model.details.n_layers:
            total_layers = model.details.n_layers
        else:
            total_layers, _, _, _ = self._get_model_architecture_details(
                model_size_b, model.details.family
            )

        if model.details.hidden_size and model.details.n_heads:
            hidden_size = model.details.hidden_size
            n_heads = model.details.n_heads
            n_kv_heads = model.details.n_kv_heads or n_heads
        else:
            _, hidden_size, n_heads, n_kv_heads = self._get_model_architecture_details(
                model_size_b, model.details.family
            )

        if n_gpu_layers == -1:
            gpu_layers_to_load = total_layers
        else:
            gpu_layers_to_load = min(n_gpu_layers, total_layers)

        # --- 3. CALCULATE MEMORY COMPONENTS ---

        # --- Component 1: Model Weights (GGUF) ---
        gpu_layer_proportion = (
            gpu_layers_to_load / total_layers if total_layers > 0 else 0
        )
        model_weights_gpu_gb = model_size_gb * gpu_layer_proportion

        # --- Component 2: KV Cache ---
        gqa_factor = n_kv_heads / n_heads if n_heads > 0 else 1
        bytes_per_token_per_layer = hidden_size * gqa_factor * 2 * 2

        # Apply efficiency factors based on real-world measurements
        # The original formula significantly overestimates KV cache usage
        if model_size_b >= 30:
            kv_efficiency = 0.15  # Large models much more efficient
        elif model_size_b >= 13:
            kv_efficiency = 0.4  # Medium models
        elif model_size_b <= 4:
            kv_efficiency = 0.8  # Small models closer to theoretical
        else:
            kv_efficiency = 0.5  # Default for 7B models

        kv_cache_bytes = (
            gpu_layers_to_load * n_ctx * bytes_per_token_per_layer * kv_efficiency
        )
        kv_cache_gb = kv_cache_bytes / (1024**3)

        # --- Component 3: Activation/Compute Buffer ---
        # Activation memory should scale with batch size and model width, not context length
        # For llama.cpp, activation memory is primarily for the forward pass computation
        # Formula: batch_size * hidden_size * num_layers_on_gpu * bytes_per_activation
        # Using n_ubatch as the effective batch size for memory calculation

        # Calculate bytes per activation based on quantization level
        if bits_per_weight >= 16:
            bytes_per_activation = 2  # FP16 activations for high precision models
        elif bits_per_weight >= 8:
            bytes_per_activation = 2  # Still use FP16 activations for 8-bit models
        else:
            bytes_per_activation = (
                1  # Can use lower precision for heavily quantized models
            )

        activation_bytes = (
            n_ubatch * hidden_size * gpu_layers_to_load * bytes_per_activation
        )
        activation_gb = activation_bytes / (1024**3)

        # --- Component 4: Overhead ---
        # Reduce overhead calculation for better accuracy
        overhead_gb = max(0.5, model_weights_gpu_gb * 0.05)

        # --- Component 5: CLIP Model (mmproj) ---
        clip_model_gb = 0.0
        if model.details.clip_model_size and model.details.clip_model_size > 0:
            clip_model_gb = model.details.clip_model_size / (1024**3)

        # --- 4. CALCULATE TOTALS ---

        total_gpu_gb = (
            model_weights_gpu_gb
            + kv_cache_gb
            + activation_gb
            + overhead_gb
            + clip_model_gb  # <-- Add CLIP model size here
        )

        # CPU memory (for layers not on GPU)
        cpu_layer_proportion = (
            (total_layers - gpu_layers_to_load) / total_layers
            if total_layers > 0
            else 0
        )
        cpu_memory_gb = model_size_gb * cpu_layer_proportion

        # Debug logging for memory breakdown
        from utils.logging import llmmllogger

        logger = llmmllogger.bind(component="Resizer")
        logger.debug(
            f"Memory breakdown: model_weights={model_weights_gpu_gb:.2f}GB, "
            f"kv_cache={kv_cache_gb:.2f}GB, activation={activation_gb:.2f}GB, "
            f"overhead={overhead_gb:.2f}GB, total={total_gpu_gb:.2f}GB "
            f"(n_ctx={n_ctx}, n_ubatch={n_ubatch}, hidden_size={hidden_size}, "
            f"gpu_layers={gpu_layers_to_load}, quant={bits_per_weight}bit, "
            f"bytes_per_activation={bytes_per_activation})"
        )

        return {
            "model_weights_gpu_gb": round(model_weights_gpu_gb, 2),
            "clip_model_gb": round(clip_model_gb, 2),  # <-- Add to breakdown
            "kv_cache_gb": round(kv_cache_gb, 2),
            "activation_gb": round(activation_gb, 2),
            "overhead_gb": round(overhead_gb, 2),
            "total_gpu_gb": round(total_gpu_gb, 2),
            "cpu_memory_gb": round(cpu_memory_gb, 2),
            "gpu_layers_loaded": gpu_layers_to_load,
            "total_layers": total_layers,
            "quantization_bits": bits_per_weight,
            "model_size_b": model_size_b,
            "model_size_gb": round(model_size_gb, 2),
            "hidden_size": hidden_size,
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
        }
