#!/usr/bin/env python3
"""
Inspect GGUF model metadata to extract architectural details.
"""

import sys
import os
import gguf

def extract_gguf_metadata(file_path):
    """Extract metadata from a GGUF file."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        reader = gguf.GGUFReader(file_path)
        
        metadata = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
        }
        
        # Extract key architectural parameters
        fields = reader.fields
        
        # Common GGUF metadata fields
        key_mappings = {
            'llama.context_length': 'original_ctx',
            'llama.block_count': 'n_layers', 
            'llama.embedding_length': 'hidden_size',
            'llama.attention.head_count': 'n_heads',
            'llama.attention.head_count_kv': 'n_kv_heads',
            'general.name': 'model_name',
            'general.architecture': 'architecture',
            'general.parameter_count': 'parameter_count',
            'general.quantization_version': 'quantization_version',
            'general.file_type': 'file_type',
            'tokenizer.ggml.model': 'tokenizer_model'
        }
        
        for gguf_key, our_key in key_mappings.items():
            if gguf_key in fields:
                value = fields[gguf_key].data
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                metadata[our_key] = value
                
        # Extract key architectural parameters
        print(f"\n📋 Key architecture fields in {os.path.basename(file_path)}:")
        important_keys = [
            'general.name', 'general.architecture', 'general.parameter_count',
            'llama.context_length', 'llama.block_count', 'llama.embedding_length',
            'llama.attention.head_count', 'llama.attention.head_count_kv',
            'llama.rope.freq_base', 'general.file_type'
        ]
        
        for key in important_keys:
            if key in fields:
                value = fields[key].data
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                if isinstance(value, list) and len(value) > 0:
                    value = value[0] if len(value) == 1 else value
                print(f"  {key}: {value}")
        
        # Store the values for return
        for key in important_keys:
            if key in fields:
                value = fields[key].data
                if isinstance(value, bytes):
                    value = value.decode('utf-8', errors='ignore')
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                metadata[key] = value
            
        return metadata
        
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_model_metadata.py <gguf_file_path>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    metadata = extract_gguf_metadata(file_path)
    
    if metadata:
        print(f"\n🔍 Extracted metadata for {os.path.basename(file_path)}:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("❌ Failed to extract metadata")

if __name__ == "__main__":
    main()