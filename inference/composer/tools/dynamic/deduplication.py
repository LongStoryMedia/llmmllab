"""
Advanced duplicate tool detection and management system for composer.
"""

import logging
import asyncio
import hashlib
import uuid
from typing import Dict, List, Optional, Set
import difflib
import ast
import re

from models import DynamicTool, ToolSimilarity, DeduplicationResult
from db import storage
from runner import pipeline_factory, Embeddings
from runner.pipeline_factory import PipelinePriority
from runner.pipelines.run import embed_pipeline


class AdvancedToolDeduplicator:
    """Advanced system for detecting and managing duplicate tools."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.similarity_threshold = 0.85  # Threshold for considering tools duplicates
        self.semantic_threshold = 0.90  # Threshold for semantic similarity

        # Cache for embeddings to avoid recomputation
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_lock = asyncio.Lock()

    async def find_similar_tools(
        self, proposed_tool: DynamicTool, conversation_ctx, limit: int = 10
    ) -> List[ToolSimilarity]:
        """Find tools similar to the proposed tool."""

        # Get embedding for the proposed tool
        proposed_embedding = await self._get_tool_embedding(
            proposed_tool, conversation_ctx
        )

        # Search for similar tools by embedding - returns tuple (tools, pagination)
        similar_tools, _ = await storage.get_service(
            storage.dynamic_tool
        ).search_user_tools_by_embedding(
            user_id=conversation_ctx.user_config.user_id,
            query_embedding=proposed_embedding,
            limit=limit * 2,  # Get more to filter by other criteria
        )

        similarities = []

        for tool in similar_tools:
            # Extract similarity score from tool (added by storage service)
            semantic_score = getattr(tool, "similarity_score", 0.0)

            # Calculate comprehensive similarity
            similarity = await self._calculate_comprehensive_similarity(
                proposed_tool, tool, semantic_score
            )
            similarities.append(similarity)

        # Sort by overall similarity score
        similarities.sort(key=lambda x: x.overall_similarity, reverse=True)

        return similarities[:limit]

    async def check_for_duplicates(
        self, proposed_tool: DynamicTool, conversation_ctx
    ) -> DeduplicationResult:
        """Check if a proposed tool is a duplicate of existing tools."""

        similar_tools = await self.find_similar_tools(
            proposed_tool, conversation_ctx, limit=5
        )

        if not similar_tools:
            return DeduplicationResult(
                is_duplicate=False,
                existing_tool=None,
                similarity_score=0.0,
                recommendation="No similar tools found. Safe to create new tool.",
                should_create_new=True,
            )

        # Check the most similar tool
        best_match = similar_tools[0]

        if best_match.overall_similarity >= self.similarity_threshold:
            existing_tool = best_match.tool

            return DeduplicationResult(
                is_duplicate=True,
                existing_tool=existing_tool,
                similarity_score=best_match.overall_similarity,
                recommendation=f"High similarity ({best_match.overall_similarity:.2f}) with existing tool '{existing_tool.name if existing_tool else 'unknown'}'. Consider reusing instead of creating new.",
                should_create_new=False,
                merge_suggestion=(
                    self._generate_merge_suggestion(proposed_tool, existing_tool)
                    if existing_tool
                    else None
                ),
            )

        return DeduplicationResult(
            is_duplicate=False,
            existing_tool=best_match.tool,
            similarity_score=best_match.overall_similarity,
            recommendation=f"Moderate similarity ({best_match.overall_similarity:.2f}) found but below threshold. Creating new tool is recommended.",
            should_create_new=True,
        )

    async def _get_tool_embedding(
        self, tool: DynamicTool, conversation_ctx
    ) -> List[float]:
        """Get embedding for a tool."""

        cache_key = self._get_cache_key(tool)

        async with self._cache_lock:
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

        # Combine tool description and code for embedding
        text_for_embedding = f"{tool.description}\n\n{tool.code}"

        mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.embedding_profile_id,
            conversation_ctx.user_config.user_id,
        )

        if not mp:
            raise ValueError("Embedding model profile not found")

        # Use LOW priority for embeddings (background task)
        with pipeline_factory.pipeline(mp, Embeddings, PipelinePriority.LOW) as pipeline:
            from typing import cast
            from runner.pipelines.base import EmbeddingPipeline
            
            embedding_result = await embed_pipeline(
                [text_for_embedding], cast(EmbeddingPipeline, pipeline)
            )

        if not embedding_result or len(embedding_result) == 0:
            raise ValueError("Failed to generate embedding for tool")

        embedding = embedding_result[0]

        # Cache the result
        async with self._cache_lock:
            self._embedding_cache[cache_key] = embedding

        return embedding

    async def _calculate_comprehensive_similarity(
        self, proposed_tool: DynamicTool, existing_tool: DynamicTool, semantic_score: float
    ) -> ToolSimilarity:
        """Calculate comprehensive similarity between two tools."""

        # Description similarity
        desc_similarity = self._calculate_text_similarity(
            proposed_tool.description, existing_tool.description
        )

        # Code similarity (structural)
        code_similarity = self._calculate_code_similarity(
            proposed_tool.code, existing_tool.code
        )

        # Function signature similarity  
        signature_similarity = self._calculate_signature_similarity(
            proposed_tool.parameters or {},
            existing_tool.parameters or {},
        )

        # Weight the different similarity measures
        overall_similarity = (
            semantic_score * 0.4
            + desc_similarity * 0.3
            + code_similarity * 0.2
            + signature_similarity * 0.1
        )

        return ToolSimilarity(
            tool=existing_tool,
            overall_similarity=overall_similarity,
            name_similarity=desc_similarity,  # Using description as name similarity for now
            description_similarity=desc_similarity,
            code_similarity=code_similarity,
            parameter_similarity=signature_similarity,
            semantic_similarity=semantic_score,
            reasons=[
                f"Semantic similarity: {semantic_score:.2f}",
                f"Description similarity: {desc_similarity:.2f}",
                f"Code similarity: {code_similarity:.2f}",
                f"Parameter similarity: {signature_similarity:.2f}",
            ],
        )

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matching."""
        return difflib.SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate structural code similarity."""
        try:
            # Parse both code snippets
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)

            # Extract structural features
            features1 = self._extract_code_features(tree1)
            features2 = self._extract_code_features(tree2)

            # Calculate similarity based on common features
            common_features = set(features1) & set(features2)
            total_features = set(features1) | set(features2)

            if not total_features:
                return 0.0

            return len(common_features) / len(total_features)

        except (SyntaxError, TypeError):
            # Fallback to text similarity if parsing fails
            return self._calculate_text_similarity(code1, code2)

    def _extract_code_features(self, tree: ast.AST) -> List[str]:
        """Extract structural features from AST."""
        features = []

        for node in ast.walk(tree):
            # Function definitions
            if isinstance(node, ast.FunctionDef):
                features.append(f"func_{node.name}")

            # Variable assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        features.append(f"var_{target.id}")

            # Function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    features.append(f"call_{node.func.id}")

            # Control structures
            elif isinstance(node, (ast.If, ast.For, ast.While)):
                features.append(f"control_{type(node).__name__.lower()}")

        return features

    def _calculate_signature_similarity(self, params1: Dict, params2: Dict) -> float:
        """Calculate similarity between function signatures."""
        if not params1 and not params2:
            return 1.0

        keys1 = set(params1.keys())
        keys2 = set(params2.keys())

        if not keys1 and not keys2:
            return 1.0

        # Calculate Jaccard similarity for parameter names
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)

        return intersection / union if union > 0 else 0.0

    def _get_cache_key(self, tool: DynamicTool) -> str:
        """Generate cache key for a tool."""
        content = f"{tool.description}{tool.code}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_merge_suggestion(
        self, proposed_tool: DynamicTool, existing_tool: Optional[DynamicTool]
    ) -> Optional[str]:
        """Generate suggestion for merging tools."""
        if not existing_tool:
            return None

        return (
            f"Consider extending existing tool '{existing_tool.name}' "
            f"instead of creating '{proposed_tool.name}'. "
            f"The existing tool already handles similar functionality."
        )