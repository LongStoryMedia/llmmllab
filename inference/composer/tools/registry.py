"""Centralized Tool Registry using internal EmbeddingAgent for semantic reuse.

This refactor removes the direct dependency on external sentence-transformer
models in favor of our unified embedding pathway (EmbeddingAgent + pipeline
factory). Responsibilities:

1. Provide static tool instances
2. Maintain registry of dynamic tools
3. Offer semantic lookup (existing vs modify/compose vs create) using
   EmbeddingAgent derived embeddings

NOTE: Dynamic tool modification and creation are still placeholders; only
      existing tool reuse path is active. Structure kept intentionally lean.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from langchain.tools import BaseTool

from models import (
    Tool,
    IntentAnalysis,
)  # DynamicTool intentionally unused pending implementation

from composer.monitoring.logging import composer_logger
from composer.core.errors import ToolGenerationError
from composer.tools.static import (
    WebSearchTool,
    MemoryRetrievalTool,
    SummarizationTool,
)
from composer.agents.embedding_agent import EmbeddingAgent

from runner import PipelineFactory


class ToolRegistry:
    """Registry & semantic selector for static + dynamic tools.

    Embedding flow:
      - Embeddings computed through EmbeddingAgent (unified model profile path)
      - Stored per tool_id for cosine similarity reuse
    """

    def __init__(
        self,
        pipeline_factory: PipelineFactory,
    ):
        # Static tool definitions (pre-defined tool classes for instantiation)
        self.static_tools: Dict[str, type[BaseTool]] = {}
        # Dynamic tool instances (id -> Tool model instances)
        self.dynamic_tools: Dict[str, Tool] = {}
        # Mapping from tool names to actual BaseTool instances for execution
        self.executable_tools: Dict[str, Any] = {}  # tool_name -> BaseTool instance
        # Semantic vectors (tool_id -> np.ndarray)
        self.tool_embeddings: Dict[str, np.ndarray] = {}

        self.pipeline_factory = pipeline_factory
        self.embedding_agent = EmbeddingAgent(pipeline_factory)
        self._lock = asyncio.Lock()

        self._load_static_tools()

    def _load_static_tools(self):
        """Load static tools from the static tools directory."""
        try:
            self.static_tools.update(
                {
                    "web_search": WebSearchTool,
                    "summarization": SummarizationTool,
                    "memory_retrieval": MemoryRetrievalTool,
                }
            )

            composer_logger.logger.info(
                "Loaded static tools", extra={"tool_count": len(self.static_tools)}
            )

        except ImportError as e:
            composer_logger.log_error(e, {"context": "static_tools_loading"})

    async def get_tools_for_context(
        self, intent: IntentAnalysis, user_id: str
    ) -> List[Tool]:
        """
        Select applicable tools based on intent and user configuration.

        Implements conditional standard tool collection and dynamic tool assessment.
        Uses shared data layer to retrieve user configuration via user_id.
        """
        tools = []

        try:
            # Get user configuration from shared data layer
            user_config = await self._get_user_config(user_id)
            tool_config = user_config.tool if user_config else None
            # Phase 1: Conditional Standard Tool Collection
            for _, tool_cls in self.static_tools.items():
                # if tool_cls and self._should_include_static_tool(name, intent):
                tool_instance = self._create_tool_instance(tool_cls, user_id)
                if tool_instance:
                    tools.append(tool_instance)

            # Phase 2: Dynamic Tool Assessment and Creation Logic
            # Check if tool generation is enabled for this intent
            tool_generation_enabled = (
                getattr(intent, "requires_tools", False)
                and tool_config
                and tool_config.enable_tool_generation
            )
            if tool_generation_enabled:
                dynamic_tool = await self._generate_or_retrieve_dynamic_tool(
                    user_id, intent
                )
                if dynamic_tool:
                    tools.append(dynamic_tool)

            composer_logger.logger.info(
                "Selected tools for context",
                extra={
                    "tool_count": len(tools),
                    "intent": intent.primary_intent,
                    "requires_tools": getattr(intent, "requires_tools", False),
                },
            )

            return tools

        except Exception as e:
            composer_logger.log_error(e, {"context": "tool_selection"})
            # Return minimal tool set on error
            return []

    async def generate_dynamic_tool(
        self, tool_spec: Any, user_id: str
    ) -> Optional[Any]:
        """
        Public interface for dynamic tool generation.

        Args:
            tool_spec: Tool specification (DynamicTool model)
            user_id: User identifier

        Returns:
            Generated tool instance or None
        """
        return await self._generate_or_retrieve_dynamic_tool(user_id, tool_spec)

    async def register_dynamic_tool_instance(
        self, tool_id: str, tool_instance: Tool, user_id: Optional[str] = None
    ) -> None:
        """
        Register a dynamic tool instance in the registry for reuse.

        Args:
            tool_id: Unique identifier for the tool
            tool_instance: The actual Tool instance to store
            user_id: Optional user id for embedding context
        """
        async with self._lock:
            self.dynamic_tools[tool_id] = tool_instance
            composer_logger.logger.info(
                f"Registered dynamic tool instance: {tool_id}",
                extra={"tool_name": getattr(tool_instance, "name", tool_id)},
            )
            # Compute & store embedding for semantic reuse if possible
            if self.embedding_agent and user_id:
                try:
                    emb = await self._compute_embedding(
                        tool_instance.description or tool_instance.name, user_id
                    )
                    if emb is not None:
                        self.tool_embeddings[tool_id] = emb
                except Exception as e:  # pragma: no cover - defensive
                    composer_logger.log_error(e, {"context": "dynamic_tool_embedding"})

    async def _get_user_config(self, user_id: str):
        """Get user configuration from shared data layer."""
        try:
            # avoid circular import
            from db import storage  # pylint: disable=import-outside-toplevel

            # from models.default_configs import DEFAULT_TOOL_CONFIG  # Currently unused

            # Initialize storage if not done
            if not storage.pool:
                composer_logger.logger.warning(
                    "Database not initialized, using default tool config"
                )
                return None

            user_config = await storage.get_service(
                storage.user_config
            ).get_user_config(user_id)
            if not user_config:
                composer_logger.logger.warning(
                    f"No user config found for {user_id}, using default tool config"
                )
                return None
            return user_config
        except Exception as e:
            composer_logger.logger.error(
                f"Failed to get user config for {user_id}: {e}, using default tool config"
            )
            return None

    async def get_static_tool_instances(self, user_id: str) -> List[Tool]:
        """
        Get instances of all static tools for a user.
        This method is for the GraphBuilder's tool collection functionality.

        Args:
            user_id: User identifier for configuration

        Returns:
            List of instantiated static Tool objects
        """
        instances = []
        for _, tool_cls in self.static_tools.items():
            if tool_cls:
                tool_instance = self._create_tool_instance(tool_cls, user_id)
                if tool_instance:
                    instances.append(tool_instance)
        return instances

    async def get_dynamic_tool_instances(self) -> List[Tool]:
        """
        Get all dynamic tool instances for a user.
        This method is for the GraphBuilder's tool collection functionality.

        Args:
            user_id: User identifier (currently not used for filtering, but available for future use)

        Returns:
            List of dynamic Tool instances
        """
        async with self._lock:
            # For now, return all dynamic tools. In the future, could filter by user_id
            return list(self.dynamic_tools.values())

    def _create_tool_instance(
        self, tool_cls: Any, user_id: str
    ) -> Optional[Tool]:
        """Create tool instance from tool class with user configuration."""
        from models import Tool as ModelTool  # Import our generic Tool model
        
        try:
            # Handle different constructor signatures for BaseTool instances
            if tool_cls.__name__ == "MemoryRetrievalTool":
                # MemoryRetrievalTool needs both user_id and conversation_id
                # Use a default conversation_id for registry - tools will be re-created with actual conversation_id at runtime
                base_tool = tool_cls(user_id=user_id, conversation_id=0)  # Default conversation_id
            else:
                # WebSearchTool and SummarizationTool need only user_id
                base_tool = tool_cls(user_id=user_id)
            
            tool_name = getattr(base_tool, 'name', tool_cls.__name__)
            
            # Store the actual BaseTool instance for execution
            self.executable_tools[tool_name] = base_tool
            
            # Convert BaseTool instance to our generic Tool model for WorkflowState compatibility
            tool_instance = ModelTool(
                name=tool_name,
                description=getattr(base_tool, 'description', f"{tool_cls.__name__} tool"),
                args_schema=getattr(base_tool, 'args_schema', None),
                return_direct=getattr(base_tool, 'return_direct', False),
                tags=getattr(base_tool, 'tags', None),
                metadata=getattr(base_tool, 'metadata', None),
                handle_tool_error=getattr(base_tool, 'handle_tool_error', False),
                handle_validation_error=getattr(base_tool, 'handle_validation_error', False),
                response_format=getattr(base_tool, 'response_format', 'content'),
            )
            
            composer_logger.logger.debug(
                "Created tool instance",
                tool_class=tool_cls.__name__,
                tool_name=tool_name,
                user_id=user_id,
                stored_executable=True,
            )
            return tool_instance
        except Exception as e:
            composer_logger.log_error(
                e, {"context": "tool_instantiation", "tool_class": str(tool_cls), "user_id": user_id}
            )
            return None

    def get_executable_tool(self, tool_name: str) -> Optional[Any]:
        """Get the actual BaseTool instance for execution by tool name."""
        return self.executable_tools.get(tool_name)

    def get_all_executable_tools(self) -> Dict[str, Any]:
        """Get all executable BaseTool instances mapped by name."""
        return self.executable_tools.copy()

    async def _generate_or_retrieve_dynamic_tool(
        self, user_id: str, intent: IntentAnalysis
    ) -> Optional[Tool]:
        """
        Implement the three-tier dynamic tool decision process:
        Use Existing -> Modify/Compose -> Create New
        """
        if not hasattr(intent, "tool_specification"):
            return None

        spec_description = getattr(intent, "tool_specification", "")

        try:
            # Compute embedding of spec description via embedding agent
            spec_embedding = await self._compute_embedding(spec_description, user_id)
            if spec_embedding is None:
                return None

            # Find similar existing tool via vector similarity
            best_match_id, similarity_score = await self._find_best_match(
                spec_embedding
            )

            composer_logger.log_tool_generation(
                tool_spec=spec_description,
                method="similarity_search",
                success=True,
                additional_context={
                    "similarity_score": similarity_score,
                    "best_match": best_match_id,
                },
            )

            # Decision logic based on similarity thresholds
            user_config = await self._get_user_config(user_id)
            tool_similarity_threshold = 0.9  # Default threshold
            tool_modification_threshold = 0.6  # Default threshold

            if user_config and user_config.tool:
                tool_similarity_threshold = getattr(
                    user_config.tool, "tool_similarity_threshold", 0.9
                )
                tool_modification_threshold = getattr(
                    user_config.tool, "tool_modification_threshold", 0.6
                )

            if best_match_id and similarity_score > tool_similarity_threshold:
                # Use Existing
                return await self._use_existing_tool(best_match_id)

            elif best_match_id and similarity_score > tool_modification_threshold:
                # Modify or Compose
                return await self._modify_or_compose_tool(best_match_id, intent)

            else:
                # Create New - temporarily disabled to avoid Tool structure issues
                # PLACEHOLDER: Implement _create_new_tool with proper Tool structure
                composer_logger.logger.warning(
                    f"Dynamic tool creation not yet implemented for user {user_id}"
                )
                return None

        except Exception as e:  # pragma: no cover - error path
            composer_logger.log_error(e, {"context": "dynamic_tool_generation"})
            raise ToolGenerationError(f"Failed to generate dynamic tool: {e}") from e

    async def _compute_embedding(self, text: str, user_id: str) -> Optional[np.ndarray]:
        """Compute embedding vector for text using EmbeddingAgent.

        Falls back to None if embedding pipeline unavailable.
        """
        if not text or not self.embedding_agent:
            return None
        try:
            vec = await self.embedding_agent.generate_single_embedding(text, user_id)
            if not vec:
                return None
            return np.array(vec, dtype=np.float32)
        except Exception as e:  # pragma: no cover - defensive
            composer_logger.log_error(e, {"context": "embedding_computation"})
            return None

    async def _find_best_match(
        self, query_embedding: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """Find tool with highest similarity to query embedding."""
        if not self.tool_embeddings:
            return None, 0.0

        best_similarity = 0.0
        best_match_id = None

        # Compute cosine similarity with all tool embeddings
        for tool_id, tool_embedding in self.tool_embeddings.items():
            similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = tool_id

        return best_match_id, best_similarity

    async def _use_existing_tool(self, tool_id: str) -> Optional[Tool]:
        """Return existing dynamic tool instance by ID."""
        async with self._lock:
            existing_tool = self.dynamic_tools.get(tool_id)
            if existing_tool:
                composer_logger.log_tool_generation(
                    tool_spec=f"existing:{tool_id}",
                    method="existing",
                    success=True,
                    tool_id=tool_id,
                )
                return existing_tool
            return None

    async def _modify_or_compose_tool(
        self, base_tool_id: str, _intent: IntentAnalysis
    ) -> Optional[Tool]:  # pragma: no cover - placeholder
        """Placeholder for modification/composition path.

        Currently returns existing tool untouched.
        """
        composer_logger.log_tool_generation(
            tool_spec=f"modify:{base_tool_id}",
            method="modified",
            success=False,
            tool_id=base_tool_id,
            additional_context={"reason": "modification_not_implemented"},
        )
        return await self._use_existing_tool(base_tool_id)

    async def _create_new_tool(
        self, _intent: IntentAnalysis, spec_description: str
    ) -> Optional[Tool]:  # pragma: no cover - placeholder
        """Placeholder for future dynamic tool creation via LLM code generation."""
        composer_logger.log_tool_generation(
            tool_spec=spec_description,
            method="new",
            success=False,
            additional_context={"reason": "creation_not_implemented"},
        )
        composer_logger.logger.warning(
            "Dynamic tool creation disabled (pending implementation)"
        )
        return None

    async def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool registry statistics."""
        async with self._lock:
            return {
                "static_tools": len([t for t in self.static_tools.values() if t]),
                "dynamic_tools": len(self.dynamic_tools),
                "total_embeddings": len(self.tool_embeddings),
                "embedding_agent_available": self.embedding_agent is not None,
            }

    async def close(self) -> None:
        """Clean up tool registry resources."""
        # Clean up any resources if needed
        self.dynamic_tools.clear()
        self.tool_embeddings.clear()
