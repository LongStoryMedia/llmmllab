"""Predefined workflow templates."""

from .chat import build_chat_workflow, get_chat_workflow_config
from .research import build_research_workflow, get_research_workflow_config

__all__ = [
    "build_chat_workflow",
    "get_chat_workflow_config", 
    "build_research_workflow",
    "get_research_workflow_config",
]
