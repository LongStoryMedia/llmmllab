"""
ToolRegistry with semantic search for dynamic tool discovery and reuse.
Implements the composab                    tool_instance = self._create_tool_instance(
                        name, tool_cls, user_id
                    )ty and abstraction requirements.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from models import Tool, IntentAnalysis, AvailableTool

from composer.monitoring.logging import composer_logger
from composer.core.errors import ToolGenerationError
from composer.tools.static import (
    WebSearchTool,
    MemoryRetrievalTool,
    SummarizationTool,
)


class ToolRegistry:
    """
    Centralized tool management with composability and reuse.

    Manages static and dynamic tools with semantic search for discovery,
    implements the three-tier decision process: Use Existing -> Modify/Compose -> Create New
    """

    def __init__(self):
        # Static tool definitions (pre-defined tools)
        self.static_tools: Dict[str, Any] = {
            "web_search": None,  # Will be loaded from static modules
            "memory_retrieval": None,
            "summarization": None,
        }

        # Dynamic tool instances (id -> tool instance)
        self.dynamic_tools: Dict[str, Any] = {}

        # Tool embeddings for semantic search (id -> embedding vector)
        self.tool_embeddings: Dict[str, np.ndarray] = {}

        # Semantic model for similarity computation
        self.embedding_model = None
        self._lock = asyncio.Lock()

        # Initialize components
        self._initialize_embedding_model()
        self._load_static_tools()

    def _initialize_embedding_model(self):
        """Initialize sentence transformer for semantic similarity."""
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            composer_logger.logger.info(
                "Initialized embedding model for tool similarity"
            )
        except Exception as e:
            composer_logger.log_error(e, {"context": "embedding_model_init"})
            # Fallback to simple text matching if embedding model fails
            self.embedding_model = None

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
    ) -> List[AvailableTool]:
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
            for name, tool_cls in self.static_tools.items():
                if tool_cls and self._should_include_static_tool(name, intent):
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

    async def _get_user_config(self, user_id: str):
        """Get user configuration from shared data layer."""
        try:
            from db import storage
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

    def _should_include_static_tool(
        self, tool_name: str, intent: IntentAnalysis
    ) -> bool:
        """Determine if a static tool should be included based on intent."""
        # Tool inclusion logic based on intent analysis
        tool_intent_mapping = {
            "web_search": ["research", "search", "information_gathering"],
            "summarization": ["research", "analysis", "content_processing"],
            "memory_retrieval": ["chat", "conversation", "context"],
        }

        relevant_intents = tool_intent_mapping.get(tool_name, [])
        return (
            intent.primary_intent in relevant_intents
            or getattr(intent, "estimated_complexity", "low") == "high"
            or getattr(intent, "requires_external_data", False)
        )

    def _create_tool_instance(self, tool_cls: Any, user_id: str) -> Optional[Tool]:  # noqa: ARG002
        """Create tool instance from tool class with user configuration."""
        # PLACEHOLDER: Use user_id to configure tool instances when needed
        try:
            # Tool instantiation logic depends on tool class interface
            # This is a simplified version - actual implementation depends on tool class structure
            return Tool(
                name=tool_cls.__name__,
                description=getattr(
                    tool_cls, "description", f"{tool_cls.__name__} tool"
                ),
            )
        except Exception as e:
            composer_logger.log_error(
                e, {"context": "tool_instantiation", "tool_class": str(tool_cls)}
            )
            return None

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
            # Compute embedding of spec description
            spec_embedding = await self._compute_embedding(spec_description)
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
                # Create New - temporarily disabled to avoid AvailableTool structure issues
                # PLACEHOLDER: Implement _create_new_tool with proper AvailableTool structure
                composer_logger.logger.warning(
                    f"Dynamic tool creation not yet implemented for user {user_id}"
                )
                return None

        except Exception as e:
            composer_logger.log_error(e, {"context": "dynamic_tool_generation"})
            raise ToolGenerationError(f"Failed to generate dynamic tool: {e}") from e

    async def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding vector for text using sentence transformer."""
        if not self.embedding_model:
            return None

        try:
            # Run embedding computation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self.embedding_model.encode, text
            )
            # Convert to numpy array if needed
            if hasattr(embedding, "numpy"):
                return embedding.numpy().astype(np.float32)
            elif hasattr(embedding, "detach"):
                return embedding.detach().cpu().numpy().astype(np.float32)
            else:
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
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
        """Return existing dynamic tool by ID."""
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
        self, base_tool_id: str, intent: IntentAnalysis  # noqa: ARG002
    ) -> Optional[Tool]:
        """Modify existing tool or compose multiple tools using LCEL."""
        # This is a placeholder for tool modification/composition logic
        # Actual implementation would use LangChain Expression Language (LCEL)
        # to compose tools as RunnableSequences

        composer_logger.log_tool_generation(
            tool_spec=f"modify:{base_tool_id}",
            method="modified",
            success=False,  # Not implemented yet
            tool_id=base_tool_id,
            additional_context={"reason": "modification_not_implemented"},
        )

        # For now, return the base tool
        return await self._use_existing_tool(base_tool_id)

    async def _create_new_tool(
        self, intent: IntentAnalysis, spec_description: str  # noqa: ARG002
    ) -> Optional[Tool]:
        """Create completely new tool using LLM code generation."""
        # This is a placeholder for new tool creation logic
        # Actual implementation would use LLM to generate tool code

        composer_logger.log_tool_generation(
            tool_spec=spec_description,
            method="new",
            success=False,  # Not implemented yet
            additional_context={"reason": "creation_not_implemented"},
        )

        # Temporarily return None to avoid AvailableTool structure issues
        # PLACEHOLDER: Implement proper dynamic tool creation with correct AvailableTool fields
        composer_logger.logger.warning("Dynamic tool creation temporarily disabled")
        return None

        # Temporarily disabled due to AvailableTool structure
        # PLACEHOLDER: Implement proper tool registration with correct AvailableTool handling
        # composer_logger.logger.warning("Tool registration temporarily disabled")
        # return

    async def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool registry statistics."""
        async with self._lock:
            return {
                "static_tools": len(
                    [t for t in self.static_tools.values() if t is not None]
                ),
                "dynamic_tools": len(self.dynamic_tools),
                "total_embeddings": len(self.tool_embeddings),
                "embedding_model_available": self.embedding_model is not None,
            }

    async def close(self) -> None:
        """Clean up tool registry resources."""
        # Clean up any resources if needed
        self.dynamic_tools.clear()
        self.tool_embeddings.clear()
