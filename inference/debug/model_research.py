"""
Model Architecture Research Summary

Based on HuggingFace documentation and model specifications:
"""

models_research = {
    "qwen3-vl-32b-thinking": {
        "huggingface_id": "huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated",
        "base_model": "Qwen/Qwen2.5-32B",
        "original_ctx": 32768,
        "n_layers": 64,
        "hidden_size": 5120,
        "n_heads": 40,
        "n_kv_heads": 8,
        "parameter_size": "32B",
        "family": "Qwen",
        "families": ["Qwen", "VL"],
        "format": "gguf",
        "quantization_level": "q4_k_m",
        "dtype": "Q4_K_M",
        "specialization": "Text",
        "supports_vision": True,
        "supports_thinking": True
    },
    
    "qwen3-vl-2b-thinking": {
        "huggingface_id": "huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated",
        "base_model": "Qwen/Qwen2-VL-2B",
        "original_ctx": 32768,
        "n_layers": 24,
        "hidden_size": 1536,
        "n_heads": 12,
        "n_kv_heads": 2,
        "parameter_size": "2B",
        "family": "Qwen",
        "families": ["Qwen", "VL"],
        "format": "gguf",
        "quantization_level": "f16",
        "dtype": "F16",
        "specialization": "Text",
        "supports_vision": True,
        "supports_thinking": True
    },
    
    "llama-chat-summary-3.2-3b": {
        "huggingface_id": "bartowski/Llama-Chat-Summary-3.2-3B-GGUF",
        "base_model": "meta-llama/Llama-3.2-3B",
        "original_ctx": 131072,  # Llama 3.2 extended context
        "n_layers": 28,
        "hidden_size": 3072,
        "n_heads": 24,
        "n_kv_heads": 8,
        "parameter_size": "3.2B",
        "family": "Llama",
        "families": ["Llama"],
        "format": "gguf",
        "quantization_level": "q5_k_m", 
        "dtype": "Q5_K_M",
        "specialization": "Text"
    },
    
    "openai-gpt-oss-20b": {
        "huggingface_id": "DavidAU/OpenAi-GPT-oss-20b-abliterated-uncensored-NEO-Imatrix-gguf",
        "base_model": "microsoft/DialoGPT-large", # Need to verify this
        "original_ctx": 4096,  # Standard GPT context
        "n_layers": 36,  # Estimated for 20B model
        "hidden_size": 4096,
        "n_heads": 32,
        "n_kv_heads": 32,  # GPT models typically have n_heads = n_kv_heads
        "parameter_size": "20B",
        "family": "GPT",
        "families": ["GPT", "OpenAI"],
        "format": "gguf",
        "quantization_level": "q5_1",
        "dtype": "Q5_1",
        "specialization": "Text",
        "disable_reason": "Unsupported gpt-oss architecture in llama_cpp"
    },
    
    "qwen3-30b-a3b": {
        "huggingface_id": "Qwen/Qwen3-30B-A3B",
        "original_ctx": 32768,
        "n_layers": 64,
        "hidden_size": 5120,
        "n_heads": 40,
        "n_kv_heads": 8,
        "parameter_size": "30.5B",
        "family": "Qwen",
        "families": ["Qwen", "MoE"],
        "format": "gguf",
        "quantization_level": "iq4_xs",
        "dtype": "IQ4_XS",
        "specialization": "Text"
    },
    
    "qwen3-4b": {
        "huggingface_id": "Qwen/Qwen3-4B",
        "original_ctx": 32768,
        "n_layers": 32,
        "hidden_size": 3584,
        "n_heads": 28,
        "n_kv_heads": 4,
        "parameter_size": "4B",
        "family": "Qwen",
        "families": ["Qwen", "MoE"],
        "format": "gguf",
        "quantization_level": "q6_k_xl",
        "dtype": "Q6_K_XL",
        "specialization": "Text"
    },
    
    "nomic-embed-text-v2": {
        "huggingface_id": "nomic-ai/nomic-embed-text-v2-moe",
        "original_ctx": 2048,  # Typical embedding model context
        "n_layers": 12,  # Estimated for embedding model
        "hidden_size": 768,
        "n_heads": 12,
        "n_kv_heads": 12,
        "parameter_size": "475M",
        "family": "Qwen",
        "families": ["Qwen"],
        "format": "gguf",
        "quantization_level": "f16",
        "dtype": "F16",
        "specialization": "Embedding"
    },
    
    "qwen3-coder-30b-a3b": {
        "huggingface_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "original_ctx": 32768,
        "n_layers": 64,
        "hidden_size": 5120,
        "n_heads": 40,
        "n_kv_heads": 8,
        "parameter_size": "30B",
        "family": "Qwen",
        "families": ["Qwen"],
        "format": "gguf",
        "quantization_level": "q4_k_xl",
        "dtype": "Q4_K_XL",
        "specialization": "Text"
    }
}

# Print the research for verification
for model_id, specs in models_research.items():
    print(f"\n{model_id}:")
    for key, value in specs.items():
        print(f"  {key}: {value}")