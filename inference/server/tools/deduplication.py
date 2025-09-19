"""
Advanced duplicate tool detection and management system.
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
from server.db import storage
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

        elif best_match.overall_similarity >= 0.7:  # Moderate similarity
            existing_tool = best_match.tool

            return DeduplicationResult(
                is_duplicate=False,
                existing_tool=existing_tool,
                similarity_score=best_match.overall_similarity,
                recommendation=f"Moderate similarity ({best_match.overall_similarity:.2f}) with existing tool. Consider if functionality can be extended instead of creating new tool.",
                should_create_new=True,  # But with caution
                merge_suggestion=(
                    self._generate_enhancement_suggestion(proposed_tool, existing_tool)
                    if existing_tool
                    else None
                ),
            )

        else:
            return DeduplicationResult(
                is_duplicate=False,
                existing_tool=None,
                similarity_score=best_match.overall_similarity,
                recommendation=f"Low similarity ({best_match.overall_similarity:.2f}) with existing tools. Safe to create new tool.",
                should_create_new=True,
            )

    async def _get_tool_embedding(
        self, tool: DynamicTool, conversation_ctx
    ) -> List[float]:
        """Get embedding for a tool, using cache if available."""

        # Create a cache key based on tool content
        tool_content = f"{tool.name} {tool.description} {' '.join(tool.parameters.keys()) if tool.parameters else ''}"
        cache_key = hashlib.md5(tool_content.encode()).hexdigest()

        async with self._cache_lock:
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]

        # Get embedding profile
        embedding_profile = await storage.get_service(
            storage.model_profile
        ).get_model_profile_by_id(
            conversation_ctx.user_config.model_profiles.embedding_profile_id,
            conversation_ctx.user_config.user_id,
        )

        if not embedding_profile:
            raise ValueError("Embedding profile not found")

        # Create message for embedding
        embedding_text = f"Tool: {tool.name}\nDescription: {tool.description}\nParameters: {tool.parameters}"

        # Get embedding
        with pipeline_factory.pipeline(
            embedding_profile, Embeddings, PipelinePriority.HIGH
        ) as pipe:
            embedding_result = await embed_pipeline(embedding_text, pipe)

            if (
                isinstance(embedding_result, list)
                and embedding_result
                and isinstance(embedding_result[0], list)
            ):
                embedding = embedding_result[0]
            else:
                raise ValueError("Invalid embedding result format")

        # Cache the result
        async with self._cache_lock:
            self._embedding_cache[cache_key] = embedding

        return embedding

    async def _calculate_comprehensive_similarity(
        self, tool1: DynamicTool, tool2: DynamicTool, semantic_score: float
    ) -> ToolSimilarity:
        """Calculate comprehensive similarity between two tools."""

        # Name similarity
        name_similarity = self._calculate_text_similarity(tool1.name, tool2.name)

        # Description similarity
        description_similarity = self._calculate_text_similarity(
            tool1.description, tool2.description
        )

        # Code similarity
        code_similarity = self._calculate_code_similarity(tool1.code, tool2.code)

        # Parameter similarity
        parameter_similarity = self._calculate_parameter_similarity(
            tool1.parameters, tool2.parameters
        )

        # Calculate weighted overall similarity
        overall_similarity = (
            name_similarity * 0.15
            + description_similarity * 0.25
            + code_similarity * 0.35
            + parameter_similarity * 0.15
            + semantic_score * 0.10
        )

        # Generate reasons for similarity
        reasons = self._generate_similarity_reasons(
            name_similarity,
            description_similarity,
            code_similarity,
            parameter_similarity,
            semantic_score,
        )

        return ToolSimilarity(
            tool=tool2,
            overall_similarity=overall_similarity,
            name_similarity=name_similarity,
            description_similarity=description_similarity,
            code_similarity=code_similarity,
            parameter_similarity=parameter_similarity,
            semantic_similarity=semantic_score,
            reasons=reasons,
        )

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        text1_norm = text1.lower().strip()
        text2_norm = text2.lower().strip()

        # Exact match
        if text1_norm == text2_norm:
            return 1.0

        # Use difflib for sequence similarity
        sequence_similarity = difflib.SequenceMatcher(
            None, text1_norm, text2_norm
        ).ratio()

        # Check for word overlap
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())

        if words1 and words2:
            word_overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            word_overlap = 0.0

        # Combine metrics
        return (sequence_similarity * 0.7) + (word_overlap * 0.3)

    def _calculate_code_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between two code blocks."""
        if not code1 or not code2:
            return 0.0

        try:
            # Normalize code by parsing and reformatting
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)

            # Extract function names, variable names, and structure
            analyzer1 = CodeAnalyzer()
            analyzer2 = CodeAnalyzer()

            analyzer1.visit(tree1)
            analyzer2.visit(tree2)

            # Compare structural elements
            function_similarity = self._compare_sets(
                analyzer1.functions, analyzer2.functions
            )
            variable_similarity = self._compare_sets(
                analyzer1.variables, analyzer2.variables
            )
            operation_similarity = self._compare_sets(
                analyzer1.operations, analyzer2.operations
            )

            # Compare raw code similarity
            normalized_code1 = self._normalize_code(code1)
            normalized_code2 = self._normalize_code(code2)
            text_similarity = self._calculate_text_similarity(
                normalized_code1, normalized_code2
            )

            # Weighted combination
            return (
                function_similarity * 0.3
                + variable_similarity * 0.2
                + operation_similarity * 0.2
                + text_similarity * 0.3
            )

        except Exception:
            # Fallback to text similarity if parsing fails
            return self._calculate_text_similarity(code1, code2)

    def _calculate_parameter_similarity(
        self, params1: Optional[Dict], params2: Optional[Dict]
    ) -> float:
        """Calculate similarity between parameter dictionaries."""
        if not params1 and not params2:
            return 1.0

        if not params1 or not params2:
            return 0.0

        keys1 = set(params1.keys())
        keys2 = set(params2.keys())

        # Key overlap
        key_overlap = (
            len(keys1.intersection(keys2)) / len(keys1.union(keys2))
            if keys1.union(keys2)
            else 0.0
        )

        # Type similarity for common keys
        type_similarities = []
        for key in keys1.intersection(keys2):
            type1 = (
                params1[key].get("type", "")
                if isinstance(params1[key], dict)
                else str(type(params1[key]).__name__)
            )
            type2 = (
                params2[key].get("type", "")
                if isinstance(params2[key], dict)
                else str(type(params2[key]).__name__)
            )
            type_similarities.append(1.0 if type1 == type2 else 0.0)

        type_similarity = (
            sum(type_similarities) / len(type_similarities)
            if type_similarities
            else 0.0
        )

        return (key_overlap * 0.7) + (type_similarity * 0.3)

    def _compare_sets(self, set1: Set[str], set2: Set[str]) -> float:
        """Compare two sets and return similarity score."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _normalize_code(self, code: str) -> str:
        """Normalize code for comparison."""
        # Remove comments and extra whitespace
        lines = []
        for line in code.split("\n"):
            # Remove comments
            line = re.sub(r"#.*$", "", line)
            # Remove extra whitespace
            line = line.strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    def _generate_similarity_reasons(
        self,
        name_sim: float,
        desc_sim: float,
        code_sim: float,
        param_sim: float,
        semantic_sim: float,
    ) -> List[str]:
        """Generate human-readable reasons for similarity scores."""
        reasons = []

        if name_sim > 0.8:
            reasons.append(f"Very similar names ({name_sim:.2f})")
        elif name_sim > 0.6:
            reasons.append(f"Similar names ({name_sim:.2f})")

        if desc_sim > 0.8:
            reasons.append(f"Nearly identical descriptions ({desc_sim:.2f})")
        elif desc_sim > 0.6:
            reasons.append(f"Similar descriptions ({desc_sim:.2f})")

        if code_sim > 0.8:
            reasons.append(f"Very similar code structure ({code_sim:.2f})")
        elif code_sim > 0.6:
            reasons.append(f"Similar code logic ({code_sim:.2f})")

        if param_sim > 0.8:
            reasons.append(f"Nearly identical parameters ({param_sim:.2f})")
        elif param_sim > 0.6:
            reasons.append(f"Similar parameters ({param_sim:.2f})")

        if semantic_sim > 0.9:
            reasons.append(f"Semantically equivalent ({semantic_sim:.2f})")
        elif semantic_sim > 0.8:
            reasons.append(f"Semantically similar ({semantic_sim:.2f})")

        if not reasons:
            reasons.append("Low overall similarity")

        return reasons

    def _generate_merge_suggestion(
        self, proposed_tool: DynamicTool, existing_tool: DynamicTool
    ) -> str:
        """Generate suggestion for merging similar tools."""
        suggestions = []

        # Compare functionality
        if len(proposed_tool.code) > len(existing_tool.code):
            suggestions.append("The proposed tool has more comprehensive functionality")
        elif len(existing_tool.code) > len(proposed_tool.code):
            suggestions.append("The existing tool is more comprehensive")

        # Compare parameters
        proposed_params = (
            set(proposed_tool.parameters.keys()) if proposed_tool.parameters else set()
        )
        existing_params = (
            set(existing_tool.parameters.keys()) if existing_tool.parameters else set()
        )

        if proposed_params - existing_params:
            suggestions.append(
                f"Proposed tool adds parameters: {', '.join(proposed_params - existing_params)}"
            )

        if existing_params - proposed_params:
            suggestions.append(
                f"Existing tool has additional parameters: {', '.join(existing_params - proposed_params)}"
            )

        if not suggestions:
            suggestions.append(
                "Tools appear functionally equivalent - use existing tool"
            )

        return "; ".join(suggestions)

    def _generate_enhancement_suggestion(
        self, proposed_tool: DynamicTool, existing_tool: DynamicTool
    ) -> str:
        """Generate suggestion for enhancing existing tool instead of creating new one."""
        suggestions = []

        # Analyze what the proposed tool adds
        proposed_words = set(proposed_tool.description.lower().split())
        existing_words = set(existing_tool.description.lower().split())

        new_concepts = proposed_words - existing_words
        if new_concepts:
            suggestions.append(
                f"Consider extending existing tool to handle: {', '.join(list(new_concepts)[:3])}"
            )

        # Check for parameter additions
        if proposed_tool.parameters and existing_tool.parameters:
            proposed_params = set(proposed_tool.parameters.keys())
            existing_params = set(existing_tool.parameters.keys())

            if proposed_params - existing_params:
                suggestions.append(
                    f"Could add parameters: {', '.join(proposed_params - existing_params)}"
                )

        if not suggestions:
            suggestions.append(
                "Consider if existing tool can be enhanced instead of creating new one"
            )

        return "; ".join(suggestions)

    async def cleanup_duplicates(
        self, user_id: str, dry_run: bool = True
    ) -> Dict[str, int]:
        """Clean up duplicate tools for a user."""
        self.logger.info(
            f"Starting duplicate cleanup for user {user_id} (dry_run={dry_run})"
        )

        # Get all tools for the user
        tools, _ = await storage.get_service(storage.dynamic_tool).list_tools_by_user(
            user_id, limit=1000
        )

        duplicates_found = 0
        tools_removed = 0
        tools_merged = 0

        # Compare each tool with others
        for i, tool1 in enumerate(tools):
            for tool2 in tools[i + 1 :]:
                # Calculate similarity without conversation context (simplified)
                similarity = await self._calculate_simple_similarity(tool1, tool2)

                if similarity > self.similarity_threshold:
                    duplicates_found += 1
                    self.logger.info(
                        f"Found duplicate: {tool1.name} <-> {tool2.name} (similarity: {similarity:.2f})"
                    )

                    if not dry_run:
                        # Decide which tool to keep (prefer newer, more comprehensive)
                        if (
                            tool1.created_at
                            and tool2.created_at
                            and tool1.created_at > tool2.created_at
                        ):
                            if tool2.id:
                                # Convert int ID to UUID if needed
                                tool2_uuid = (
                                    uuid.UUID(int=tool2.id)
                                    if isinstance(tool2.id, int)
                                    else tool2.id
                                )
                                await storage.get_service(
                                    storage.dynamic_tool
                                ).delete_tool(tool2_uuid, user_id)
                                tools_removed += 1
                        else:
                            if tool1.id:
                                # Convert int ID to UUID if needed
                                tool1_uuid = (
                                    uuid.UUID(int=tool1.id)
                                    if isinstance(tool1.id, int)
                                    else tool1.id
                                )
                                await storage.get_service(
                                    storage.dynamic_tool
                                ).delete_tool(tool1_uuid, user_id)
                                tools_removed += 1

        return {
            "duplicates_found": duplicates_found,
            "tools_removed": tools_removed,
            "tools_merged": tools_merged,
        }

    async def _calculate_simple_similarity(
        self, tool1: DynamicTool, tool2: DynamicTool
    ) -> float:
        """Calculate simplified similarity without embeddings."""
        name_sim = self._calculate_text_similarity(tool1.name, tool2.name)
        desc_sim = self._calculate_text_similarity(tool1.description, tool2.description)
        code_sim = self._calculate_code_similarity(tool1.code, tool2.code)

        return name_sim * 0.2 + desc_sim * 0.3 + code_sim * 0.5


class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze code structure."""

    def __init__(self):
        self.functions = set()
        self.variables = set()
        self.operations = set()

    def visit_FunctionDef(self, node):
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.variables.add(node.id)
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.operations.add(type(node.op).__name__)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.functions.add(node.func.id)
        self.generic_visit(node)


# Global deduplicator instance
tool_deduplicator = AdvancedToolDeduplicator()
