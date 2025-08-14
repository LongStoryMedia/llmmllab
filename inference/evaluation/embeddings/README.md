# Embedding Model Evaluation

This module provides tools for evaluating embedding models, specifically the Nomic Embed Text v2 model. It allows you to:

1. Generate embeddings from text using proper instruction prefixes
2. Compare the similarity between pairs of texts
3. Evaluate embedding quality against expected similarity ranges
4. Test with different Matryoshka embedding dimensions (768, 512, or 256)

## Components

- `embedding_benchmark.py`: Benchmark class for evaluating embedding models
- `extractor.py`: Direct embedding extraction using llama-cpp-python
- `utils.py`: Utility functions for working with embeddings
- `adapter.py`: Simplified adapter for direct embedding generation
- `examples/`: Example scripts showing how to use the evaluation tools

## Features

### Simplified Implementation

This version uses a simplified implementation that:

- Directly uses llama-cpp-python without any dependencies on runner or langchain
- Works as a standalone module without external dependencies
- Properly handles instruction prefixes and normalization
- Supports Matryoshka dimensions

## Requirements

```
numpy>=1.24.0
llama-cpp-python>=0.2.0
```

## Usage

### Running the Embedding Evaluation

To run the example evaluation script:

```bash
cd /path/to/llmmllab
python -m inference.evaluation.embeddings.run_embedding_example --model_path /path/to/nomic-embed-text-v2.gguf
```

Options:

- `--model-path`: Path to GGUF model file (required)
- `--data-dir`: Directory containing test data (default: "./benchmark_data/embeddings")
- `--test-file`: Test file to use (default: "basic_embedding_test.json")
- `--output`: Output file to save results (default: "embedding_evaluation_results.json")
- `--matryoshka-dim`: Matryoshka dimension for truncated embeddings (256, 512, or 768)

### Comparing Text Similarity

To compare the similarity between two texts:

```bash
cd /path/to/llmmllab
python -m inference.evaluation.embeddings.examples.compare_similarity \
  --model-path /path/to/nomic-embed-text-v2.gguf \
  --text1 "What is machine learning?" \
  --text2 "Machine learning is a field of AI" \
  --is-query1
```

Options:

- `--model`: Model ID (default: "nomic-embed-text-v2")
- `--text1`: First text to compare
- `--text2`: Second text to compare
- `--is-query1`: Treat text1 as a query
- `--is-query2`: Treat text2 as a query
- `--file1`: File containing first text
- `--file2`: File containing second text
- `--matryoshka-dim`: Matryoshka dimension (256, 512, or 768)

## Creating Custom Test Cases

You can create custom test cases by creating a JSON file in the following format:

```json
[
  {
    "query": "What is machine learning?",
    "documents": [
      "Machine learning is a branch of artificial intelligence.",
      "Deep learning is a subset of machine learning.",
      "Quantum computing uses quantum mechanics principles."
    ],
    "expected_similarities": [
      {"text": "Machine learning is a branch of artificial intelligence.", "min": 0.7, "max": 1.0},
      {"text": "Deep learning is a subset of machine learning.", "min": 0.6, "max": 1.0},
      {"text": "Quantum computing uses quantum mechanics principles.", "min": 0.1, "max": 0.5}
    ]
  }
]
```

## Notes on Embedding Prefixes

The Nomic Embed Text v2 model requires specific instruction prefixes:

- Use `search_query:` for queries/questions
- Use `search_document:` for documents/content

These prefixes are automatically added by the evaluation tools if not present in your input text.
