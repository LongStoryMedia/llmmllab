#!/usr/bin/env python3
"""Quick GGUF header inspector for comparing model & mmproj files.

Usage (in container/pod):
  python -m inference.debug.inspect_gguf_headers /models/qwen3-vl-32b/qwen3-vl-32b-q4-k-m-thinking-abliterated.gguf
  python -m inference.debug.inspect_gguf_headers /models/qwen3-vl-32b/mmproj-model-f16.gguf

Displays: magic, version, tensor count, metadata keys, truncation warnings.
This helps identify corrupt or mismatched mmproj vs quantized model artifacts.
"""
import sys
import struct
from pathlib import Path

# GGUF magic constants (ascii 'GGUF')
MAGIC = b'GGUF'


def read_header(path: Path):
    with path.open('rb') as f:
        magic = f.read(4)
        if magic != MAGIC:
            return {"file": str(path), "error": "Bad magic", "magic": magic}
        # version (uint32 little endian)
        ver_bytes = f.read(4)
        if len(ver_bytes) < 4:
            return {"file": str(path), "error": "Truncated version"}
        (version,) = struct.unpack('<I', ver_bytes)
        # tensor count (uint64)
        tensor_cnt_bytes = f.read(8)
        if len(tensor_cnt_bytes) < 8:
            return {"file": str(path), "error": "Truncated tensor count"}
        (tensor_count,) = struct.unpack('<Q', tensor_cnt_bytes)
        # metadata kv pairs count (uint64)
        meta_cnt_bytes = f.read(8)
        if len(meta_cnt_bytes) < 8:
            return {"file": str(path), "error": "Truncated meta count"}
        (kv_count,) = struct.unpack('<Q', meta_cnt_bytes)
        # For brevity, just read keys lengths without full decode
        keys = []
        for _ in range(min(kv_count, 50)):  # cap to 50
            # key length (uint64)
            len_bytes = f.read(8)
            if len(len_bytes) < 8:
                break
            (klen,) = struct.unpack('<Q', len_bytes)
            k = f.read(klen)
            if len(k) < klen:
                break
            keys.append(k.decode('utf-8', errors='replace'))
            # skip value type (uint8)
            vtype = f.read(1)
            if not vtype:
                break
            # We won't parse full value; skip naive
            # NOTE: Proper GGUF value parsing depends on type code
            # This quick inspector only lists keys.
        return {
            "file": str(path),
            "version": version,
            "tensor_count": tensor_count,
            "kv_count": kv_count,
            "sample_keys": keys,
        }


def main():
    if len(sys.argv) < 2:
        print("Provide one or more GGUF file paths.")
        return 1
    for p in sys.argv[1:]:
        path = Path(p)
        if not path.exists():
            print(f"❌ {p}: file does not exist")
            continue
        info = read_header(path)
        if 'error' in info:
            print(f"❌ {info['file']}: {info['error']} (magic={info.get('magic')})")
        else:
            print(f"📄 {info['file']} | ver={info['version']} tensors={info['tensor_count']} kv={info['kv_count']}")
            print("   keys:", ", ".join(info['sample_keys'][:10]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
