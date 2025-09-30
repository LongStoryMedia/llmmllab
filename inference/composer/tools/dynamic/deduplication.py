"""
Advanced duplicate tool detection and management system for composer.
"""

import re
import json
import logging
import asyncio
import hashlib
import difflib
import ast
from typing import Dict, List, Optional, cast

from models import (
    DynamicTool,
    ToolSimilarity,
    DeduplicationResult,
    ModelProfileType,
    PipelinePriority,
)
from db import storage
from runner import (
    pipeline_factory,
    Embeddings,
    embed_pipeline,
    run_pipeline,
    EmbeddingPipeline,
)
from utils.model_profile import get_model_profile_for_task
from utils.grammar_generator import parse_structured_output


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
        self, proposed_tool: DynamicTool, user_id: str, limit: int = 10
    ) -> List[ToolSimilarity]:
        """Find tools similar to the proposed tool."""

        # Get embedding for the proposed tool
        proposed_embedding = await self._get_tool_embedding(proposed_tool, user_id)

        # Search for similar tools by embedding - returns tuple (tools, pagination)
        similar_tools, _ = await storage.get_service(
            storage.dynamic_tool
        ).search_user_tools_by_embedding(
            user_id=user_id,
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
        self, proposed_tool: DynamicTool, user_id: str
    ) -> DeduplicationResult:
        """Check if a proposed tool is a duplicate of existing tools."""

        similar_tools = await self.find_similar_tools(proposed_tool, user_id, limit=5)

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

    async def _get_tool_embedding(self, tool: DynamicTool, user_id: str) -> List[float]:
        """Get embedding for a tool."""

        cache_key = self._get_cache_key(tool)

        async with self._cache_lock:
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

        # Combine tool description and code for embedding
        text_for_embedding = f"{tool.description}\n\n{tool.code}"

        # Retrieve user configuration from shared data layer
        uc = await storage.get_service(storage.user_config).get_user_config(user_id)
        if not uc:
            raise ValueError(f"User configuration not found for user {user_id}")

        # Get model profile for embedding task
        mp = await get_model_profile_for_task(
            uc.model_profiles, ModelProfileType.Embedding, user_id
        )

        if not mp:
            raise ValueError("Embedding model profile not found")

        # Use LOW priority for embeddings (background task)
        with pipeline_factory.pipeline(
            mp, Embeddings, PipelinePriority.LOW
        ) as pipeline:
            embedding_result = await embed_pipeline(
                text_for_embedding, cast(EmbeddingPipeline, pipeline)
            )

        if not embedding_result or len(embedding_result) == 0:
            raise ValueError("Failed to generate embedding for tool")

        embedding = embedding_result[0]

        # Cache the result
        async with self._cache_lock:
            self._embedding_cache[cache_key] = embedding

        return embedding

    async def _calculate_comprehensive_similarity(
        self,
        proposed_tool: DynamicTool,
        existing_tool: DynamicTool,
        semantic_score: float,
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

    async def analyze_tools_with_structured_output(
        self,
        proposed_tool: DynamicTool,
        existing_tools: List[DynamicTool],
        user_id: str,
    ) -> DeduplicationResult:
        """
        Perform advanced deduplication analysis using grammar-constrained structured output.

        This method leverages LLM analysis with guaranteed structured output to provide
        more sophisticated duplicate detection and merging recommendations.
        """
        # Retrieve user configuration from shared data layer
        uc = await storage.get_service(storage.user_config).get_user_config(user_id)
        if not uc:
            raise ValueError(f"User configuration not found for user {user_id}")

        # Get model profile for analysis task
        mp = await get_model_profile_for_task(
            uc.model_profiles, ModelProfileType.Analysis, user_id
        )

        if not mp:
            raise ValueError("Analysis model profile not found")

        # Create structured analysis prompt
        analysis_prompt = self._create_deduplication_analysis_prompt(
            proposed_tool, existing_tools
        )

        # Execute grammar-constrained analysis using DeduplicationResult schema
        with pipeline_factory.pipeline(
            mp, str, PipelinePriority.NORMAL, mp.circuit_breaker
        ) as pipeline:
            # Use DeduplicationResult as grammar constraint for structured output
            result = await run_pipeline(
                analysis_prompt, pipeline, grammar=DeduplicationResult
            )

            if (
                result
                and result.message
                and result.message.content
                and result.message.content[0].text
            ):
                analysis_text = result.message.content[0].text
                try:
                    # With grammar constraints, the output should be valid JSON matching DeduplicationResult

                    return parse_structured_output(analysis_text, DeduplicationResult)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to parse grammar-constrained output, falling back: {e}"
                    )
                    return self._parse_structured_deduplication_result(
                        analysis_text, existing_tools
                    )
            else:
                # Fallback to basic similarity analysis
                return await self.check_for_duplicates(proposed_tool, user_id)

    def _create_deduplication_analysis_prompt(
        self, proposed_tool: DynamicTool, existing_tools: List[DynamicTool]
    ) -> str:
        """Create structured prompt for deduplication analysis."""

        existing_tools_info = []
        for i, tool in enumerate(existing_tools[:5]):  # Limit to top 5 for context
            existing_tools_info.append(
                f"Tool {i+1}:\n"
                f"  Name: {tool.name}\n"
                f"  Description: {tool.description}\n"
                f"  Code Preview: {tool.code[:200]}...\n"
            )

        return f"""
Analyze the proposed tool against existing tools to determine if it's a duplicate and provide recommendations.

PROPOSED TOOL:
Name: {proposed_tool.name}
Description: {proposed_tool.description}
Code: {proposed_tool.code}

EXISTING SIMILAR TOOLS:
{chr(10).join(existing_tools_info)}

Analyze for:
1. Functional overlap and redundancy
2. Code similarity and reusability  
3. Potential for merging or extending existing tools
4. Recommendation for creating new vs reusing existing

Provide analysis in the following JSON structure:
{{
  "is_duplicate": boolean,
  "similarity_score": number (0.0-1.0),
  "recommendation": "string describing recommended action with detailed explanation",
  "should_create_new": boolean,
  "merge_suggestion": "string with specific merge guidance or null"
}}

Focus on functional equivalence rather than syntactic similarity.
"""

    def _parse_structured_deduplication_result(
        self, analysis_text: str, existing_tools: List[DynamicTool]
    ) -> DeduplicationResult:
        """Parse structured analysis result with fallback handling."""
        try:
            # Extract JSON from response (handle potential markdown formatting)
            json_match = re.search(r"\{.*\}", analysis_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON structure found in response")

            analysis_data = json.loads(json_match.group())

            # Find the best matching existing tool if duplicate detected
            existing_tool = None
            if analysis_data.get("is_duplicate", False) and existing_tools:
                existing_tool = existing_tools[0]  # Use first/most similar tool

            return DeduplicationResult(
                is_duplicate=analysis_data.get("is_duplicate", False),
                existing_tool=existing_tool,
                similarity_score=float(analysis_data.get("similarity_score", 0.0)),
                recommendation=analysis_data.get(
                    "recommendation", "Analysis completed"
                ),
                should_create_new=analysis_data.get("should_create_new", True),
                merge_suggestion=analysis_data.get("merge_suggestion"),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.warning(f"Failed to parse structured deduplication result: {e}")

            # Fallback to heuristic analysis
            return DeduplicationResult(
                is_duplicate=False,
                existing_tool=existing_tools[0] if existing_tools else None,
                similarity_score=0.5 if existing_tools else 0.0,
                recommendation=f"Structured analysis failed, using heuristic fallback. Consider manual review. Error: {e}",
                should_create_new=True,
            )
