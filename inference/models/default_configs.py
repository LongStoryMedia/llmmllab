"""
Default configuration objects for user settings.

This file contains predefined default objects for each configuration type.
When schemas are updated and models regenerated, the linter will indicate
where these defaults need to be updated.
"""

from models.preferences_config import PreferencesConfig
from models.memory_config import MemoryConfig
from models.summarization_config import SummarizationConfig
from models.refinement_config import RefinementConfig
from models.web_search_config import WebSearchConfig
from models.image_generation_config import ImageGenerationConfig
from models.model_profile_config import ModelProfileConfig
from models.user_config import UserConfig, WebSearchProviders

from models.default_model_profiles import DEFAULT_MODEL_PROFILE_CONFIG


# Default preferences configuration
DEFAULT_PREFERENCES_CONFIG = PreferencesConfig(
    theme="light", language="en", notifications_on=True, font_size=14
)

# Default memory configuration
DEFAULT_MEMORY_CONFIG = MemoryConfig(
    enabled=True,
    limit=100,
    enable_cross_user=False,
    enable_cross_conversation=True,
    similarity_threshold=0.7,
    always_retrieve=False,
)

# Default summarization configuration
DEFAULT_SUMMARIZATION_CONFIG = SummarizationConfig(
    enabled=True,
    messages_before_summary=10,
    summaries_before_consolidation=3,
    embedding_dimension=1536,
    max_summary_levels=3,
    summary_weight_coefficient=0.5,
)

# Default refinement configuration
DEFAULT_REFINEMENT_CONFIG = RefinementConfig(
    enable_response_filtering=True, enable_response_critique=True
)

# Default web search configuration
DEFAULT_WEB_SEARCH_CONFIG = WebSearchConfig(
    enabled=True,
    auto_detect=True,
    max_results=5,
    include_results=True,
    search_providers=[
        WebSearchProviders.GOOGLE,
        WebSearchProviders.BRAVE,
        WebSearchProviders.SERPER,
        # WebSearchProviders.SEARX,
        WebSearchProviders.DDG,
    ],
    max_urls_deep=3,
)

# Default image generation configuration
DEFAULT_IMAGE_GENERATION_CONFIG = ImageGenerationConfig(
    enabled=True,
    storage_directory="images",
    max_image_size=2048,
    retention_hours=72,
    auto_prompt_refinement=True,
    width=1024,
    height=1024,
    inference_steps=50,
    guidance_scale=7.5,
    low_memory_mode=False,
    negative_prompt="blurry, distorted, low quality, pixelated",
)


# Function to create a default user config
def create_default_user_config(user_id: str) -> UserConfig:
    """Create a default user configuration with predefined defaults for all settings"""
    return UserConfig(
        user_id=user_id,
        preferences=DEFAULT_PREFERENCES_CONFIG,
        memory=DEFAULT_MEMORY_CONFIG,
        summarization=DEFAULT_SUMMARIZATION_CONFIG,
        refinement=DEFAULT_REFINEMENT_CONFIG,
        web_search=DEFAULT_WEB_SEARCH_CONFIG,
        image_generation=DEFAULT_IMAGE_GENERATION_CONFIG,
        model_profiles=DEFAULT_MODEL_PROFILE_CONFIG,
    )
