#!/usr/bin/env python3
"""
Debug script to test metadata extraction from GGUF files.
"""

import sys
sys.path.insert(0, '/opt/venv/shared/lib/python3.12/site-packages')
import gguf
import os

# Find all GGUF files
all_files = []
for root, dirs, files in os.walk('/models'):
    for f in files:
        if f.endswith('.gguf'):
            all_files.append(os.path.join(root, f))

for path in sorted(all_files):
    try:
        reader = gguf.GGUFReader(path)

        # Extract metadata using the same method as the main script
        metadata = {}
        for key in reader.fields.keys():
            field = reader.fields[key]
            value = field.contents()
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            metadata[key] = value

        arch = metadata.get('general.architecture', '').lower()
        name = metadata.get('general.name', os.path.basename(path))
        size = os.path.getsize(path)

        print(f"File: {path}")
        print(f"  Architecture: {arch}")
        print(f"  Name: {name}")
        print(f"  Size: {size}")

        # Try to extract architecture-specific values
        block_count = (
            metadata.get('llama.block_count') or
            metadata.get('phi3.block_count') or
            metadata.get('qwen3next.block_count') or
            metadata.get('qwen35moe.block_count') or
            metadata.get('clip.vision.block_count') or
            metadata.get('nomic-bert-moe.block_count')
        )
        context_length = (
            metadata.get('llama.context_length') or
            metadata.get('phi3.context_length') or
            metadata.get('qwen3next.context_length') or
            metadata.get('qwen35moe.context_length') or
            metadata.get('clip.vision.context_length') or
            metadata.get('general.context_length') or
            2048
        )
        embedding_length = (
            metadata.get('llama.embedding_length') or
            metadata.get('phi3.embedding_length') or
            metadata.get('qwen3next.embedding_length') or
            metadata.get('qwen35moe.embedding_length') or
            metadata.get('clip.vision.embedding_length') or
            metadata.get('nomic-bert-moe.embedding_length')
        )

        print(f"  n_layers (block_count): {block_count}")
        print(f"  context_length: {context_length}")
        print(f"  hidden_size (embedding_length): {embedding_length}")
        print()
    except Exception as e:
        print(f"Error reading {path}: {e}")
        print()