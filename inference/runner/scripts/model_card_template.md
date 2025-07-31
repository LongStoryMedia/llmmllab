# {{name}}

{{description}}

## Model Information

This is a GGUF quantized model for use with [llama.cpp](https://github.com/ggerganov/llama.cpp) and compatible libraries.

### Details

- **Format:** {{format}}
- **Quantization:** {{quantization}}
- **File:** {{file.name}}
- **File Size:** {{file.size_mb}} MB
- **MD5 Hash:** {{file.md5_hash}}
- **Upload Date:** {{upload_date}}

## Usage

### With llama.cpp

```bash
./main -m /path/to/{{file.name}} -n 1024 -p "Your prompt here"
```

### With Python (llama-cpp-python)

```python
from llama_cpp import Llama

# Initialize the model
model = Llama(
    model_path="/path/to/{{file.name}}",
    n_ctx=4096,  # Context window size
    n_gpu_layers=-1  # Use as many GPU layers as possible
)

# Generate text
output = model.create_completion(
    prompt="Your prompt here",
    max_tokens=1024,
    temperature=0.7,
    top_p=0.95
)

print(output["choices"][0]["text"])
```

## License

This model is subject to the original model license.
