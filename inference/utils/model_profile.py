"""
Shared utility for model profile management.

This module provides utilities to retrieve model profiles based on task types
and user configurations.
"""

from typing import Dict
import uuid

from models.model_profile_config import ModelProfileConfig
from models.model_profile_type import ModelProfileType
from models.model_profile import ModelProfile

# Database import moved to function level to avoid circular dependencies


# Mapping between ModelProfileType enum and ModelProfileConfig field names
PROFILE_TYPE_TO_CONFIG_FIELD: Dict[ModelProfileType, str] = {
    ModelProfileType.Primary: "primary_profile_id",
    ModelProfileType.PrimarySummary: "summarization_profile_id",
    ModelProfileType.MasterSummary: "master_summary_profile_id",
    ModelProfileType.BriefSummary: "brief_summary_profile_id",
    ModelProfileType.KeyPoints: "key_points_profile_id",
    ModelProfileType.SelfCritique: "self_critique_profile_id",
    ModelProfileType.Improvement: "improvement_profile_id",
    ModelProfileType.MemoryRetrieval: "memory_retrieval_profile_id",
    ModelProfileType.Analysis: "analysis_profile_id",
    ModelProfileType.ResearchTask: "research_task_profile_id",
    ModelProfileType.ResearchPlan: "research_plan_profile_id",
    ModelProfileType.ResearchConsolidation: "research_consolidation_profile_id",
    ModelProfileType.ResearchAnalysis: "research_analysis_profile_id",
    ModelProfileType.Embedding: "embedding_profile_id",
    ModelProfileType.Formatting: "formatting_profile_id",
    ModelProfileType.ImageGenerationPrompt: "image_generation_prompt_profile_id",
    ModelProfileType.Engineering: "engineering_profile_id",
    ModelProfileType.Reranking: "reranking_profile_id",
    ModelProfileType.ImageGeneration: "image_generation_profile_id",
}


async def get_model_profile(user_id: str, task: ModelProfileType) -> ModelProfile:
    """
    Get the appropriate model profile for a specific task.

    Args:
        user_id: The user ID for profile lookup
        task: The type of task requiring a model profile

    Returns:
        The model profile for the specified task

    Raises:
        ValueError: If the task type is not supported or profile not found
    """
    from db import storage  # pylint: disable=import-outside-toplevel

    config = await storage.get_service(storage.user_config).get_user_config(user_id)
    if not config:
        raise ValueError(f"User config not found for user {user_id}")

    return await get_model_profile_for_task(config.model_profiles, task, user_id)


async def get_model_profile_for_task(
    config: ModelProfileConfig, task: ModelProfileType, user_id: str
) -> ModelProfile:
    """
    Get the appropriate model profile for a specific task.

    Args:
        config: The user's model profile configuration
        task: The type of task requiring a model profile
        user_id: The user ID for profile lookup

    Returns:
        The model profile for the specified task

    Raises:
        ValueError: If the task type is not supported or profile not found
        AssertionError: If the retrieved profile is None
    """
    # Get the profile ID from the config
    profile_id: uuid.UUID = get_profile_id_for_task(config, task)

    # Retrieve the model profile from storage
    from db import storage  # pylint: disable=import-outside-toplevel

    mp = await storage.get_service(storage.model_profile).get_model_profile_by_id(
        profile_id, user_id
    )

    assert mp, f"No profile for {task} (ID: {profile_id}, User: {user_id})"

    return mp


def get_profile_id_for_task(
    config: ModelProfileConfig, task: ModelProfileType
) -> uuid.UUID:
    """
    Get the profile ID for a specific task without database lookup.

    Args:
        config: The user's model profile configuration
        task: The type of task requiring a model profile

    Returns:
        The UUID of the profile for the specified task

    Raises:
        ValueError: If the task type is not supported
    """
    config_field = PROFILE_TYPE_TO_CONFIG_FIELD.get(task)
    if config_field is None:
        raise ValueError(f"Unsupported task type: {task}")

    return getattr(config, config_field)
