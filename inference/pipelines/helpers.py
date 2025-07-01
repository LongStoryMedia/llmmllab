

from typing import Optional
import torch

from models.model import Model


def get_dtype(model: Model) -> torch.dtype:
    """
    Factory function to return the appropriate data type string.
    """
    dtype = model.details.dtype.lower() if model.details and model.details.dtype else "float32"
    if dtype in ["float16", "fp16"]:
        return torch.float16
    elif dtype in ["bfloat16", "bfp16"]:
        return torch.bfloat16
    elif dtype in ["float32", "fp32"]:
        return torch.float32
    else:
        print(f"WARNING! Unsupported dtype '{dtype}' for model {model.name}. Defaulting to float32.")
        return torch.float32


def get_precision(model: Model) -> Optional[str]:
    """
    Factory function to return the appropriate precision string.
    """
    if not model.details or not model.details.dtype:
        return None

    dtype = model.details.dtype.lower()
    if dtype in ["float16", "fp16"]:
        return "fp16"
    elif dtype in ["bfloat16", "bfp16"]:
        return "bfp16"
    elif dtype in ["float32", "fp32"]:
        return "fp32"
    else:
        print(f"WARNING! Unsupported dtype '{dtype}' for model {model.name}. Defaulting to None.")
        return None
