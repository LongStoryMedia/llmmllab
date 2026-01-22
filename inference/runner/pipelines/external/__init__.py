"""
External pipeline implementations for API-based models.
"""

from .openai_pipeline import OpenAIPipeline

__all__ = ["OpenAIPipeline"]

# Conditionally import Anthropic if available
try:
    from .anthropic_pipeline import AnthropicPipeline

    __all__.append("AnthropicPipeline")
except ImportError:
    pass
