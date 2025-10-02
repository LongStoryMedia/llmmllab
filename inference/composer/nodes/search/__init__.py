"""
Search depth routing components for adaptive information retrieval in agentic systems.
Implements shallow/deep search routing based on query complexity and intent classification.
"""

from .router import SearchDepthRouter, ShallowSearchExecutor, DeepSearchExecutor

__all__ = [
    "SearchDepthRouter",
    "ShallowSearchExecutor", 
    "DeepSearchExecutor",
]