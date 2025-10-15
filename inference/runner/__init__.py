"""
Runner Service Interface Layer.

Provides the public API boundary for the runner component, enabling other
services to execute model pipelines while maintaining strict architectural
decoupling. This interface abstracts pipeline execution, streaming, and
embedding generation.

Interface Functions:
- pipeline_factory(): Create configured pipeline instances
- run_pipeline(): Execute pipelines with conversation context
- stream_pipeline(): Stream-enabled pipeline execution
- embed_pipeline(): Generate embeddings from text content

Interface Types:
- PipeReturn: Pipeline execution result type
- Embeddings: Embedding vector type definitions
- EmbeddingPipeline: Pipeline interface for embedding generation

Architectural Role:
- Defines clean API boundaries between components
- Abstracts internal runner implementation details
- Enables Protocol-based dependency injection
- Maintains decoupling from server and composer components
"""

__version__ = "0.1.0"

from .pipeline_factory import pipeline_factory, PipeReturn, PipelineFactory

from .pipelines.base import Embeddings, EmbeddingPipeline  # for typing
from .pipelines.run import (
    run_pipeline,
    stream_pipeline,
    embed_pipeline,
    GrammarInput,
    MessageInput,
)


__all__ = [
    "pipeline_factory",
    "PipeReturn",
    "Embeddings",
    "run_pipeline",
    "stream_pipeline",
    "embed_pipeline",
    "EmbeddingPipeline",
    "GrammarInput",
    "MessageInput",
    "PipelineFactory",
]
