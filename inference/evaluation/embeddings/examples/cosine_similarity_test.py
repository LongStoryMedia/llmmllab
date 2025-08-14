#!/usr/bin/env python
"""
Simple utility for testing cosine similarity between vectors.

This script demonstrates how to compute the cosine similarity between vectors
using NumPy implementation.
"""

import numpy as np
import torch


# Define a utility function
def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


assert cosine_similarity([1, 0, 0], [0, 1, 0]) == 0.0
assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1.0
assert np.isclose(cosine_similarity([1, 1, 0], [1, 0, 0]), 1 / np.sqrt(2))

# Tests with random vectors
v1 = np.random.rand(768)
v2 = np.random.rand(768)
v1_norm = v1 / np.linalg.norm(v1)
v2_norm = v2 / np.linalg.norm(v2)

# Test with PyTorch tensors
t1 = torch.tensor(v1_norm, dtype=torch.float32)
t2 = torch.tensor(v2_norm, dtype=torch.float32)

torch_sim = torch.nn.functional.cosine_similarity(
    t1.unsqueeze(0), t2.unsqueeze(0)
).item()
numpy_sim = cosine_similarity(v1_norm, v2_norm)

print(f"PyTorch similarity: {torch_sim:.6f}")
print(f"NumPy similarity: {numpy_sim:.6f}")
assert np.isclose(torch_sim, numpy_sim)

print("All tests passed!")
