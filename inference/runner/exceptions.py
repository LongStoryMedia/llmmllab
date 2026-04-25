from typing import Dict, List


class InsufficientVRAMError(Exception):
    """Raised when a model cannot be loaded due to insufficient VRAM.

    Attributes:
        required_bytes: Estimated VRAM needed by the incoming model.
        loaded_models: List of currently cached models with their size and lock status.
    """

    def __init__(self, required_bytes: float, loaded_models: List[Dict]):
        self.required_bytes = required_bytes
        self.loaded_models = loaded_models
        model_list = ", ".join(
            f"{m['name']} ({m['size_gb']:.1f}GB{' locked' if m.get('in_use') else ''})"
            for m in loaded_models
        )
        msg = (
            f"Insufficient VRAM for model requiring {required_bytes / 1e9:.1f}GB. "
            f"Loaded models: {model_list}. Try again later."
        )
        super().__init__(msg)
