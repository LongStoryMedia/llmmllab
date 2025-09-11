"""
Utility functions for LlamaCpp-based pipelines.
"""


def calculate_optimal_gpu_layers(
    n_ctx: int, model_size_category: str = "medium"
) -> int:
    """
    Dynamically calculate optimal GPU layers based on context size and model category.

    Args:
        n_ctx: Context size in tokens
        model_size_category: Model size category ("small", "medium", "large", "xlarge")
            - "small": ~3B-7B models (e.g., 3B embeddings)
            - "medium": ~13B-20B models (e.g., OpenAI GPT OSS 20B)
            - "large": ~30B-40B models (e.g., Qwen3-Coder 30B)
            - "xlarge": ~70B+ models

    Strategy:
    - Higher context = fewer GPU layers (more KV cache in GPU VRAM)
    - Lower context = more GPU layers (model weights in GPU VRAM)
    - Larger models need more conservative GPU allocation
    """
    # Define base allocations per model size (AGGRESSIVE for research/testing with 48GB VRAM)
    base_allocations = {
        "small": {"max": 99, "high": 95, "medium": 90, "low": 85, "min": 80},
        "medium": {"max": 95, "high": 90, "medium": 85, "low": 80, "min": 75},
        "large": {
            "max": 90,
            "high": 85,
            "medium": 80,
            "low": 75,
            "min": 70,
        },  # Much more aggressive for 30B
        "xlarge": {"max": 85, "high": 80, "medium": 75, "low": 70, "min": 65},
    }

    # Get allocation thresholds for the model category
    alloc = base_allocations.get(model_size_category, base_allocations["medium"])

    # Determine GPU layers based on context size (AGGRESSIVE allocation)
    if n_ctx <= 4096:  # 4K context or less
        return alloc["max"]  # Maximum layers on GPU
    elif n_ctx <= 8192:  # 8K context
        return alloc["high"]  # High allocation
    elif n_ctx <= 16384:  # 16K context
        return alloc["medium"]  # Still aggressive
    elif n_ctx <= 32768:  # 32K context
        return alloc["low"]  # Reasonable allocation
    elif n_ctx <= 65536:  # 64K context
        return alloc["min"]  # Still good allocation
    elif n_ctx <= 131072:  # 128K context (model's trained context)
        return max(
            20, alloc["min"] - 5
        )  # Slightly more conservative but still reasonable
    elif n_ctx <= 262144:  # 256K context
        return max(18, alloc["min"] - 7)  # More conservative for larger context
    elif n_ctx <= 524288:  # 512K context
        return max(15, alloc["min"] - 10)  # Very conservative for large context
    else:  # > 512K context (1M tokens)
        return max(12, alloc["min"] - 13)  # Minimal layers for extreme context
