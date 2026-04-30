"""
Main ComposerService orchestrator.
Central to the redesign - serves as the primary, authoritative execution runtime.

Configuration Management:
- Configuration overrides and default merging happens at the data layer
- Configuration is NOT passed as arguments in composer components
- Allowed arguments: user_id, messages/query, tools, workflow_type
- Components retrieve configuration from shared data layer using user_id
- No configuration merging logic should exist in service layer components
"""

from typing import Dict, Optional, Type

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from transformers import ModelCard

from models import (
    Message,
    MessageRole,
    MessageContent,
    MessageContentType,
)

from graph.workflows.base import GraphBuilder
from graph.state import WorkflowState
from graph.cache import WorkflowCache
from graph.executor import WorkflowExecutor
from utils.logging import llmmllogger


class ComposerService:
    """
    Main composer service coordinating graph construction and execution.

    The Composer is responsible for:
    - Graph construction & execution
    - Streaming orchestration
    - State management
    - Tool management
    - Intent analysis
    - Error resiliency
    - Multi-agent orchestration
    """

    def __init__(self, builder: GraphBuilder):
        self.logger = llmmllogger.bind(component="ComposerService")
        self.graph_builder = builder
        # Workflow cache is now created per-user during workflow composition
        self.workflow_caches: Dict[str, WorkflowCache] = {}
        # Generic workflow executor for streaming
        self.executor = WorkflowExecutor()

    async def compose_workflow(
        self,
        user_id: str,
        model_name: Optional[str] = None,
        response_format: Optional[Type[BaseModel]] = None,
        **build_kwargs,
    ) -> CompiledStateGraph:
        """
        Construct or retrieve a master workflow with intelligent routing.

        The workflow will handle intent analysis, tool selection, and routing
        internally using LangGraph's native capabilities.

        args:
            user_id: User ID for configuration retrieval

        returns:
            CompiledStateGraph: Master workflow with intelligent routing
        """
        try:
            # 1. Get user configuration from shared data layer
            from db import storage  # pylint: disable=import-outside-toplevel

            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)

            # 2. Use per-user cache if enabled (cache based on user_id only now)
            user_cache = None
            if user_config.workflow.enable_workflow_caching:
                if user_id not in self.workflow_caches:
                    self.workflow_caches[user_id] = WorkflowCache()
                user_cache = self.workflow_caches[user_id]

                # Simplified cache key - master workflow is the same for all users
                cache_key = f"workflow_{user_id}"

                if model_name:
                    cache_key += f"_{model_name}"
                    cached_workflow = await user_cache.get(cache_key)
                else:
                    entry = user_cache.cache.popitem()[1]
                    assert entry is not None, "Cached workflow should not be None"
                    cached_workflow = entry.access()

                if cached_workflow:
                    self.logger.debug(
                        "Retrieved workflow from cache",
                        extra={"cache_key": cache_key},
                    )
                    return cached_workflow

            # 3. Build master workflow
            assert self.graph_builder is not None, "GraphBuilder should be initialized"

            # Filter out None-valued kwargs so empty tool params don't bypass cache
            effective_kwargs = {k: v for k, v in build_kwargs.items() if v is not None}

            if user_cache and not effective_kwargs:
                # Only use cache when no dynamic kwargs (tools change per request)
                workflow = await user_cache.get_or_create(
                    cache_key,
                    lambda: self.graph_builder.build_workflow(
                        user_id, response_format, model_name=model_name, **build_kwargs
                    ),
                )
            else:
                workflow = await self.graph_builder.build_workflow(
                    user_id, response_format, model_name=model_name, **build_kwargs
                )

            self.logger.info(
                "Master workflow composed successfully", extra={"user_id": user_id}
            )

            return workflow

        except Exception as e:
            self.logger.error(
                "Failed to compose master workflow",
                extra={"error": str(e), "user_id": user_id},
                exc_info=True,
            )
            raise

    async def shutdown(self):
        """Clean up resources on service shutdown."""
        self.logger.info("Shutting down ComposerService")

        # Close all per-user workflow caches
        for user_id, cache in self.workflow_caches.items():
            try:
                await cache.close()
            except Exception as e:
                self.logger.warning(f"Error closing cache for user {user_id}: {e}")
        self.workflow_caches.clear()
        # Close other resources as needed
