"""
Search depth routing components for adaptive information retrieval in agentic systems.
Implements shallow/deep search routing based on query complexity and intent classification.
"""

from .router import ResearchRouter, QuickResearchExecutor, ComprehensiveResearchExecutor

__all__ = [
    "ResearchRouter",
    "QuickResearchExecutor",
    "ComprehensiveResearchExecutor",
]
