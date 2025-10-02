"""Predefined workflow templates."""

from .chat import build_chat_workflow, get_chat_workflow_config
from .research import build_research_workflow, get_research_workflow_config
from .multi_agent import build_multi_agent_workflow, get_multi_agent_workflow_config
from .creative import build_creative_workflow, get_creative_workflow_config

__all__ = [
    "build_chat_workflow",
    "get_chat_workflow_config", 
    "build_research_workflow",
    "get_research_workflow_config",
    "build_multi_agent_workflow",
    "get_multi_agent_workflow_config",
    "build_creative_workflow",
    "get_creative_workflow_config",
]
