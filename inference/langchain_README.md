# LangChain Integration

This module provides a simple integration between LangChain and our inference system. It allows you to use LangChain components with our LLM API server.

## Requirements

Install the required packages with:

```bash
pip install -r langchain_requirements.txt
```

This will install:
- langchain
- langchain-core
- langchain-openai
- openai
- requests

## Usage

### Basic Usage

```python
import os
from inference.langchain_integration import get_client, is_available

# Check if LangChain is available
if not is_available():
    print("LangChain is not available. Install with: pip install -r langchain_requirements.txt")
    exit(1)

# Create a LangChain client
client = get_client(
    api_base=os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1"),
    api_key="EMPTY"  # Not needed for local vLLM servers
)

# Create a chat model
chat_model = client.get_chat_model(
    model_name="llama2-7b",  # Use your model name
    temperature=0.7,
    max_tokens=1024
)

# Use the model
response = chat_model.invoke("Tell me a story about a robot learning to paint.")
print(response.content)
```

### Async Usage

```python
import asyncio

async def main():
    chat_model = client.get_chat_model("llama2-7b")
    response = await chat_model.ainvoke("Tell me a story about a robot learning to paint.")
    print(response.content)

asyncio.run(main())
```

## Examples

See the `examples/langchain_example.py` file for a complete example of how to use the LangChain integration.

## Advanced Usage

Once you have a LangChain model, you can use all of LangChain's features, including:

- Chains
- Agents
- Memory
- Retrieval

For more information, see the [LangChain documentation](https://python.langchain.com/docs/get_started/introduction).
