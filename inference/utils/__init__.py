"""
Shared utility modules for the inference system.
"""

from .model_profile import get_model_profile_for_task, get_profile_id_for_task

__all__ = [
    "get_model_profile_for_task",
    "get_profile_id_for_task",
]