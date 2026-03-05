"""
Backward-compatible alias for AgentNode.
PassthroughNode is now merged into AgentNode with optional tool_registry.
"""

from composer.graph.nodes.agent import AgentNode as PassthroughNode

__all__ = ["PassthroughNode"]
