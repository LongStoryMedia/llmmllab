"""
Default model profiles for the inference system.
"""

from unittest.mock import DEFAULT
import uuid
from datetime import datetime
from typing import Dict, Optional, List

from .model_profile import ModelProfile
from .model_parameters import ModelParameters
from .model_profile_config import ModelProfileConfig
from .model_profile_image_settings import ModelProfileImageSettings

# Define profile types as constants (similar to the Go implementation)
MODEL_PROFILE_TYPE_PRIMARY = 1
MODEL_PROFILE_TYPE_PRIMARY_SUMMARY = 2
MODEL_PROFILE_TYPE_MASTER_SUMMARY = 3
MODEL_PROFILE_TYPE_BRIEF_SUMMARY = 4
MODEL_PROFILE_TYPE_KEY_POINTS = 5
MODEL_PROFILE_TYPE_SELF_CRITIQUE = 6
MODEL_PROFILE_TYPE_IMPROVEMENT = 7
MODEL_PROFILE_TYPE_MEMORY_RETRIEVAL = 8
MODEL_PROFILE_TYPE_ANALYSIS = 9
MODEL_PROFILE_TYPE_RESEARCH_TASK = 10
MODEL_PROFILE_TYPE_RESEARCH_PLAN = 11
MODEL_PROFILE_TYPE_RESEARCH_CONSOLIDATION = 12
MODEL_PROFILE_TYPE_RESEARCH_ANALYSIS = 13
MODEL_PROFILE_TYPE_EMBEDDING = 14
MODEL_PROFILE_TYPE_FORMATTING = 15
MODEL_PROFILE_TYPE_IMAGE_GENERATION_PROMPT = 16
MODEL_PROFILE_TYPE_IMAGE_GENERATION = 17
MODEL_PROFILE_TYPE_ENGINEERING = 18
MODEL_PROFILE_TYPE_RERANKING = 19

# Create default UUIDs (similar to the Go implementation)
DEFAULT_PRIMARY_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_SUMMARIZATION_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
DEFAULT_MASTER_SUMMARY_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000003")
DEFAULT_BRIEF_SUMMARY_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000004")
DEFAULT_KEY_POINTS_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000005")
DEFAULT_SELF_CRITIQUE_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000006")
DEFAULT_IMPROVEMENT_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000007")
DEFAULT_MEMORY_RETRIEVAL_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000008")
DEFAULT_ANALYSIS_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000009")
DEFAULT_RESEARCH_TASK_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000010")
DEFAULT_RESEARCH_PLAN_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000011")
DEFAULT_RESEARCH_CONSOLIDATION_PROFILE_ID = uuid.UUID(
    "00000000-0000-0000-0000-000000000012"
)
DEFAULT_RESEARCH_ANALYSIS_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000013")
DEFAULT_EMBEDDING_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000014")
DEFAULT_FORMATTING_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000015")
DEFAULT_IMAGE_GENERATION_PROMPT_PROFILE_ID = uuid.UUID(
    "00000000-0000-0000-0000-000000000016"
)
DEFAULT_IMAGE_GENERATION_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000017")
DEFAULT_ENGINEERING_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000018")
DEFAULT_RERANKING_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000019")

# Default model IDs from models.json
DEFAULT_TEXT_TO_TEXT_MODEL = "qwen3-30b-a3b-q4-k-m"
DEFAULT_VISION_TEXT_TO_TEXT_MODEL = "qwen2.5-vl-32b-instruct-q4-k-m"
DEFAULT_TEXT_TO_IMAGE_MODEL = "black-forest-labs-flux.1-dev"
DEFAULT_IMAGE_TO_IMAGE_MODEL = "black-forest-labs-flux.1-kontext-dev"
DEFAULT_TEXT_TO_EMBEDDINGS_MODEL = "nomic-embed-text-v2"
DEFAULT_SUMMARIZATION_MODEL = "llama-chat-summary-3_2-3b-q5-k-m"
DEFAULT_ANALYSIS_MODEL = "qwen3-4b-ud-q6-k-xl"

# Define default model profiles
DEFAULT_PRIMARY_PROFILE = ModelProfile(
    id=DEFAULT_PRIMARY_PROFILE_ID,
    user_id="system",
    name="Primary (Default)",
    type=MODEL_PROFILE_TYPE_PRIMARY,
    description="Primary model profile for general chat and reasoning.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=40960,  # Restore original working context
        repeat_last_n=128,
        repeat_penalty=1.1,
        temperature=0.65,
        seed=-1,
        num_predict=-1,
        top_k=30,
        top_p=0.95,
        min_p=0.05,
        max_tokens=16384,  # Restore original working tokens
        n_parts=-1,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
        ],
        think=True,
    ),
    system_prompt="""You are a helpful AI assistant. When thinking through problems:

CRITICAL THINKING GUIDELINES:
- Keep your reasoning concise and focused (max 2-3 short paragraphs)
- Avoid repeating the same logic or analysis multiple times
- If you find yourself restating similar points, STOP and provide your answer
- Do not elaborate on the same concept repeatedly
- Make your thinking efficient and direct

RESPONSE STRUCTURE:
1. Brief analysis (if needed)
2. Direct, clear answer
3. Move on immediately

Avoid circular reasoning, excessive elaboration, or repetitive explanations. Be decisive and concise.""",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_SUMMARIZATION_PROFILE = ModelProfile(
    id=DEFAULT_SUMMARIZATION_PROFILE_ID,
    user_id="system",
    name="Summarization (Default)",
    type=MODEL_PROFILE_TYPE_PRIMARY_SUMMARY,
    description="Default profile for conversation summarization.",
    model_name=DEFAULT_SUMMARIZATION_MODEL,
    parameters=ModelParameters(
        num_ctx=131072,  # Increased context for summarization
        repeat_last_n=128,  # Increased for better repetition detection
        repeat_penalty=1.15,  # Higher penalty to prevent repetition
        temperature=0.1,  # Very low temperature for focused summaries
        seed=0,
        num_predict=4096,  # Increased max tokens for comprehensive synthesis
        top_k=30,  # Reduced for more focused output
        top_p=0.85,  # Reduced for less randomness
        min_p=0.05,
        max_tokens=4096,  # Increased max tokens for comprehensive synthesis
        n_parts=-1,
        stop=[
            "</s>",
            "<|endoftext|>",
            "<|end|>",
            "[/INST]",
            "\n\nSummary:",
            "\n\nText:",
            "Additionally,",
            "Furthermore,",
            "Moreover,",
        ],
        think=False,
        batch_size=64,
    ),
    system_prompt="Summarize the conversation so far in a concise paragraph. Include key points and conclusions, but omit redundant details. Be brief and focused.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_MASTER_SUMMARY_PROFILE = ModelProfile(
    id=DEFAULT_MASTER_SUMMARY_PROFILE_ID,
    user_id="system",
    name="Master Summary (Default)",
    type=MODEL_PROFILE_TYPE_MASTER_SUMMARY,
    description="Profile for generating master summaries.",
    model_name=DEFAULT_SUMMARIZATION_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.3,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
        think=False,
    ),
    system_prompt="Create a comprehensive summary of the conversation, giving most weight to the most recent points and less to older information.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_BRIEF_SUMMARY_PROFILE = ModelProfile(
    id=DEFAULT_BRIEF_SUMMARY_PROFILE_ID,
    user_id="system",
    name="Brief Summary (Default)",
    type=MODEL_PROFILE_TYPE_BRIEF_SUMMARY,
    description="Profile for generating brief summaries.",
    model_name=DEFAULT_SUMMARIZATION_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.2,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
    ),
    system_prompt="Create a very concise summary of these short messages. Focus only on essential information and be extremely brief.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_KEY_POINTS_PROFILE = ModelProfile(
    id=DEFAULT_KEY_POINTS_PROFILE_ID,
    user_id="system",
    name="Key Points (Default)",
    type=MODEL_PROFILE_TYPE_KEY_POINTS,
    description="Profile for extracting key points from messages.",
    model_name=DEFAULT_SUMMARIZATION_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.2,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.6,
        min_p=0.0,
        think=False,
    ),
    system_prompt="Extract and list the key points from these detailed messages. Identify the main ideas and important details, organizing them in a clear structure.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_SELF_CRITIQUE_PROFILE = ModelProfile(
    id=DEFAULT_SELF_CRITIQUE_PROFILE_ID,
    user_id="system",
    name="Self Critique (Default)",
    type=MODEL_PROFILE_TYPE_SELF_CRITIQUE,
    description="Profile for self-critique and response evaluation.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.4,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
    ),
    system_prompt="You are an expert critique assistant. Your task is to analyze the following AI response and identify:\n"
    "1. Factual inaccuracies or potential errors\n"
    "2. Areas where clarity could be improved\n"
    "3. Opportunities to make the response more helpful or comprehensive\n"
    "4. Any redundancies or unnecessary content\n"
    "Be concise and focus on actionable feedback that can improve the response.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_IMPROVEMENT_PROFILE = ModelProfile(
    id=DEFAULT_IMPROVEMENT_PROFILE_ID,
    user_id="system",
    name="Improvement (Default)",
    type=MODEL_PROFILE_TYPE_IMPROVEMENT,
    description="Profile for improving and refining responses.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.4,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
    ),
    system_prompt="Your task is to improve the original AI response based on the critique provided. "
    "Maintain the overall structure and intent of the original response, but address the issues identified in the critique. "
    "The improved response should be clear, accurate, concise, and directly answer the user's original query.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_MEMORY_RETRIEVAL_PROFILE = ModelProfile(
    id=DEFAULT_MEMORY_RETRIEVAL_PROFILE_ID,
    user_id="system",
    name="Memory Retrieval (Default)",
    type=MODEL_PROFILE_TYPE_MEMORY_RETRIEVAL,
    description="Profile for retrieving and summarizing memory/context.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.2,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
        think=False,
    ),
    system_prompt="Retrieve relevant information from memory and present it concisely.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_ANALYSIS_PROFILE = ModelProfile(
    id=DEFAULT_ANALYSIS_PROFILE_ID,
    user_id="system",
    name="Analysis (Default)",
    type=MODEL_PROFILE_TYPE_ANALYSIS,
    description="Profile for detailed analysis of text.",
    model_name=DEFAULT_ANALYSIS_MODEL,
    parameters=ModelParameters(
        num_ctx=40960,
        repeat_last_n=-1,
        repeat_penalty=1.05,
        temperature=0.7,
        seed=0,
        num_predict=-1,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        max_tokens=16384,
        n_parts=-1,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
        ],
        think=False,
    ),
    system_prompt="Perform an in-depth analysis of the provided text. Identify key themes, patterns, and insights.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_RESEARCH_TASK_PROFILE = ModelProfile(
    id=DEFAULT_RESEARCH_TASK_PROFILE_ID,
    user_id="system",
    name="Research Task (Default)",
    type=MODEL_PROFILE_TYPE_RESEARCH_TASK,
    description="Profile for research task generation.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.4,
        seed=0,
        num_predict=-1,
        top_k=50,
        top_p=0.9,
        min_p=0.05,
        think=False,
    ),
    system_prompt="Generate specific research tasks based on the research goals. Each task should be focused, actionable, and help address the overall research objective.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_RESEARCH_PLAN_PROFILE = ModelProfile(
    id=DEFAULT_RESEARCH_PLAN_PROFILE_ID,
    user_id="system",
    name="Research Plan (Default)",
    type=MODEL_PROFILE_TYPE_RESEARCH_PLAN,
    description="Profile for research planning.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.3,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
    ),
    system_prompt="Create a detailed research plan that outlines the steps needed to investigate this topic thoroughly. Include specific questions to explore and potential sources of information.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_RESEARCH_CONSOLIDATION_PROFILE = ModelProfile(
    id=DEFAULT_RESEARCH_CONSOLIDATION_PROFILE_ID,
    user_id="system",
    name="Research Consolidation (Default)",
    type=MODEL_PROFILE_TYPE_RESEARCH_CONSOLIDATION,
    description="Profile for consolidating research findings.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=8192,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.2,
        seed=0,
        num_predict=-1,
        top_k=40,
        top_p=0.9,
        min_p=0.0,
    ),
    system_prompt="Consolidate the research findings into a coherent summary. Identify common themes, highlight key insights, and note any conflicts or gaps in the information.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_RESEARCH_ANALYSIS_PROFILE = ModelProfile(
    id=DEFAULT_RESEARCH_ANALYSIS_PROFILE_ID,
    user_id="system",
    name="Research Analysis (Default)",
    type=MODEL_PROFILE_TYPE_RESEARCH_ANALYSIS,
    description="Profile for analyzing research results.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=40960,
        repeat_last_n=-1,
        repeat_penalty=1.05,
        temperature=0.7,
        seed=0,
        num_predict=-1,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        max_tokens=16384,
        n_parts=-1,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
        ],
    ),
    system_prompt="Analyze the research findings critically. Evaluate the strength of evidence, identify potential biases, and suggest areas for further investigation.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_EMBEDDING_PROFILE = ModelProfile(
    id=DEFAULT_EMBEDDING_PROFILE_ID,
    user_id="system",
    name="Embedding (Default)",
    type=MODEL_PROFILE_TYPE_EMBEDDING,
    description="Profile for generating text embeddings.",
    model_name=DEFAULT_TEXT_TO_EMBEDDINGS_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        temperature=0.0,  # No randomness for embeddings
        seed=0,
    ),
    system_prompt="Generate high-quality vector embeddings for the input text.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_RERANKING_PROFILE = ModelProfile(
    id=DEFAULT_RERANKING_PROFILE_ID,
    user_id="system",
    name="Content Re-ranking (Default)",
    type=MODEL_PROFILE_TYPE_RERANKING,
    description="Profile for re-ranking and de-duplicating search results.",
    model_name="rerank-content",  # This is a virtual model ID that will be mapped to the ReRankPipeline
    parameters=ModelParameters(
        num_ctx=2048,
        temperature=0.0,  # No randomness for re-ranking
        seed=0,
    ),
    system_prompt="Re-rank and deduplicate search results based on relevance to the query.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_FORMATTING_PROFILE = ModelProfile(
    id=DEFAULT_FORMATTING_PROFILE_ID,
    user_id="system",
    name="Formatting (Default)",
    type=MODEL_PROFILE_TYPE_FORMATTING,
    description="Profile for text formatting and structure.",
    model_name=DEFAULT_ANALYSIS_MODEL,
    parameters=ModelParameters(
        num_ctx=40960,
        repeat_last_n=-1,
        repeat_penalty=1.05,
        temperature=0.7,
        seed=0,
        num_predict=-1,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        max_tokens=16384,
        n_parts=-1,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
        ],
        think=False,
    ),
    system_prompt="Format the provided text according to best practices. Improve structure, organization, and readability while preserving all content.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_IMAGE_GENERATION_PROMPT_PROFILE = ModelProfile(
    id=DEFAULT_IMAGE_GENERATION_PROMPT_PROFILE_ID,
    user_id="system",
    name="Image Generation Prompt (Default)",
    type=MODEL_PROFILE_TYPE_IMAGE_GENERATION_PROMPT,
    description="Profile for generating prompts for image generation.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,
    parameters=ModelParameters(
        num_ctx=2048,
        repeat_last_n=64,
        repeat_penalty=1.1,
        temperature=0.7,
        seed=0,
        num_predict=-1,
        top_k=50,
        top_p=0.95,
        min_p=0.05,
    ),
    system_prompt="Convert the user's image request into a detailed, high-quality prompt for image generation. Include specific details about style, composition, lighting, and content.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_IMAGE_GENERATION_PROFILE = ModelProfile(
    id=DEFAULT_IMAGE_GENERATION_PROFILE_ID,
    user_id="system",
    name="Image Generation (Default)",
    type=MODEL_PROFILE_TYPE_IMAGE_GENERATION,
    description="Profile for image generation.",
    model_name=DEFAULT_TEXT_TO_IMAGE_MODEL,
    parameters=ModelParameters(
        num_ctx=1024,
        temperature=1.0,
        seed=0,
    ),
    system_prompt="Generate high-quality images based on the provided prompt.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

DEFAULT_ENGINEERING_PROFILE = ModelProfile(
    id=DEFAULT_ENGINEERING_PROFILE_ID,
    user_id="system",
    name="Engineering (Default)",
    type=MODEL_PROFILE_TYPE_ENGINEERING,
    description="Profile for engineering tasks.",
    model_name="qwen3-coder-30b-a3b",
    parameters=ModelParameters(
        num_ctx=100000,
        repeat_last_n=-1,
        repeat_penalty=1.05,
        temperature=0.7,
        seed=0,
        num_predict=-1,
        top_k=20,
        top_p=0.8,
        min_p=0.0,
        max_tokens=16384,
        n_parts=-1,
        stop=[
            "<|im_end|>",
            "<|endoftext|>",
            "<|end|>",
        ],
        batch_size=256,
    ),
    system_prompt="Assist with engineering tasks, providing detailed explanations and solutions.",
    created_at=datetime.now(),
    updated_at=datetime.now(),
)

# Create the default model profile config
DEFAULT_MODEL_PROFILE_CONFIG = ModelProfileConfig(
    primary_profile_id=DEFAULT_PRIMARY_PROFILE_ID,
    summarization_profile_id=DEFAULT_SUMMARIZATION_PROFILE_ID,
    master_summary_profile_id=DEFAULT_MASTER_SUMMARY_PROFILE_ID,
    brief_summary_profile_id=DEFAULT_BRIEF_SUMMARY_PROFILE_ID,
    key_points_profile_id=DEFAULT_KEY_POINTS_PROFILE_ID,
    improvement_profile_id=DEFAULT_IMPROVEMENT_PROFILE_ID,
    memory_retrieval_profile_id=DEFAULT_MEMORY_RETRIEVAL_PROFILE_ID,
    self_critique_profile_id=DEFAULT_SELF_CRITIQUE_PROFILE_ID,
    analysis_profile_id=DEFAULT_ANALYSIS_PROFILE_ID,
    research_task_profile_id=DEFAULT_RESEARCH_TASK_PROFILE_ID,
    research_plan_profile_id=DEFAULT_RESEARCH_PLAN_PROFILE_ID,
    research_consolidation_profile_id=DEFAULT_RESEARCH_CONSOLIDATION_PROFILE_ID,
    research_analysis_profile_id=DEFAULT_RESEARCH_ANALYSIS_PROFILE_ID,
    embedding_profile_id=DEFAULT_EMBEDDING_PROFILE_ID,
    formatting_profile_id=DEFAULT_FORMATTING_PROFILE_ID,
    image_generation_prompt_profile_id=DEFAULT_IMAGE_GENERATION_PROMPT_PROFILE_ID,
    image_generation_profile_id=DEFAULT_IMAGE_GENERATION_PROFILE_ID,
    engineering_profile_id=DEFAULT_ENGINEERING_PROFILE_ID,
    reranking_profile_id=DEFAULT_RERANKING_PROFILE_ID,
)


# Helper function to get a model by task
def get_model_for_task(task: str) -> str:
    """Get the best model ID for a specific task from models.json"""
    import json
    import os

    # Default mappings in case models.json can't be loaded
    default_models = {
        "TextToText": DEFAULT_TEXT_TO_TEXT_MODEL,
        "VisionTextToText": DEFAULT_VISION_TEXT_TO_TEXT_MODEL,
        "TextToImage": DEFAULT_TEXT_TO_IMAGE_MODEL,
        "ImageToImage": DEFAULT_IMAGE_TO_IMAGE_MODEL,
        "TextToEmbeddings": DEFAULT_TEXT_TO_EMBEDDINGS_MODEL,
    }

    try:
        # Attempt to load models.json
        models_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models.json"
        )
        with open(models_path, "r") as f:
            models_data = json.load(f)

        # Find the first model matching the task
        for model in models_data:
            if model.get("task") == task:
                return model.get("id")

        # Fall back to default if no match found
        return default_models.get(task, DEFAULT_TEXT_TO_TEXT_MODEL)
    except Exception:
        # If there's an error loading models.json, return the default
        return default_models.get(task, DEFAULT_TEXT_TO_TEXT_MODEL)


# Create a mapping of all default profiles
DEFAULT_PROFILES = {
    "primary": DEFAULT_PRIMARY_PROFILE,
    "summarization": DEFAULT_SUMMARIZATION_PROFILE,
    "master_summary": DEFAULT_MASTER_SUMMARY_PROFILE,
    "brief_summary": DEFAULT_BRIEF_SUMMARY_PROFILE,
    "key_points": DEFAULT_KEY_POINTS_PROFILE,
    "improvement": DEFAULT_IMPROVEMENT_PROFILE,
    "memory_retrieval": DEFAULT_MEMORY_RETRIEVAL_PROFILE,
    "self_critique": DEFAULT_SELF_CRITIQUE_PROFILE,
    "analysis": DEFAULT_ANALYSIS_PROFILE,
    "research_task": DEFAULT_RESEARCH_TASK_PROFILE,
    "research_plan": DEFAULT_RESEARCH_PLAN_PROFILE,
    "research_consolidation": DEFAULT_RESEARCH_CONSOLIDATION_PROFILE,
    "research_analysis": DEFAULT_RESEARCH_ANALYSIS_PROFILE,
    "embedding": DEFAULT_EMBEDDING_PROFILE,
    "formatting": DEFAULT_FORMATTING_PROFILE,
    "image_generation_prompt": DEFAULT_IMAGE_GENERATION_PROMPT_PROFILE,
    "image_generation": DEFAULT_IMAGE_GENERATION_PROFILE,
    "reranking": DEFAULT_RERANKING_PROFILE,
    "engineering": DEFAULT_ENGINEERING_PROFILE,
}
