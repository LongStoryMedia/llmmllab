# Model Profile Configuration Guide

## Overview

The LLM ML Lab uses a hierarchical model profile configuration system that allows users to customize model behavior for different tasks while providing sensible defaults. The system consists of **default system profiles**, **user configuration overrides**, and **individual profile customization**.

## Architecture

### 1. Default System Profiles

Located in: `inference/models/default_model_profiles.py`

The system defines 19 default model profiles for different task types:

| Profile Type | Default Model | Purpose |
|-------------|---------------|---------|
| **Primary** | `qwen3-30b-a3b-q4-k-m` | General chat and reasoning |
| **Summarization** | `llama-chat-summary-3_2-3b-q5-k-m` | Text summarization |
| **Master Summary** | `qwen3-30b-a3b-q4-k-m` | Comprehensive summaries |
| **Brief Summary** | `qwen3-30b-a3b-q4-k-m` | Concise summaries |
| **Key Points** | `qwen3-30b-a3b-q4-k-m` | Bullet point extraction |
| **Analysis** | `qwen3-30b-a3b-q4-k-m` | Analytical reasoning |
| **Research Task** | `qwen3-30b-a3b-q4-k-m` | Research planning |
| **Engineering** | `qwen3-coder-30b-a3b` | Code generation and debugging |
| **Embedding** | `nomic-embed-text-v2` | Text embeddings |
| **Image Generation** | `black-forest-labs-flux.1-dev` | Image creation |

### 2. Default Model Constants

```python
# Core model assignments
DEFAULT_TEXT_TO_TEXT_MODEL = "qwen3-30b-a3b-q4-k-m"
DEFAULT_VISION_TEXT_TO_TEXT_MODEL = "qwen2.5-vl-32b-instruct-q4-k-m"
DEFAULT_TEXT_TO_IMAGE_MODEL = "black-forest-labs-flux.1-dev"
DEFAULT_IMAGE_TO_IMAGE_MODEL = "black-forest-labs-flux.1-kontext-dev"
DEFAULT_TEXT_TO_EMBEDDINGS_MODEL = "nomic-embed-text-v2"
DEFAULT_SUMMARIZATION_MODEL = "llama-chat-summary-3_2-3b-q5-k-m"
```

### 3. Profile Configuration Structure

Each model profile contains:

- **ID**: Unique UUID identifier
- **Model Name**: Reference to a model in `.models.json`
- **Parameters**: Model-specific parameters (temperature, context size, etc.)
- **System Prompt**: Task-specific instructions
- **Type**: Profile type enum (Primary, Analysis, etc.)

### 4. User Configuration Overrides

Users can override default profile assignments through `UserConfig.model_profiles`:

```python
class ModelProfileConfig(BaseModel):
    primary_profile_id: uuid.UUID
    summarization_profile_id: uuid.UUID
    master_summary_profile_id: uuid.UUID
    brief_summary_profile_id: uuid.UUID
    # ... additional profiles
```

## Configuration Flow

1. **System Initialization**: Default profiles are created in the database
2. **User Registration**: User gets default `ModelProfileConfig`
3. **Profile Selection**: System uses `primary_profile_id` for main tasks
4. **Task Routing**: Different tasks use specific profile types
5. **User Customization**: Users can create custom profiles and update config

## Primary Model Configuration

### Current Setup

The **primary model profile** is already configured to use `"qwen3-30b-a3b-q4-k-m"` as specified in your requirements:

```python
DEFAULT_PRIMARY_PROFILE = ModelProfile(
    id=DEFAULT_PRIMARY_PROFILE_ID,
    user_id="system",
    name="Primary (Default)",
    type=MODEL_PROFILE_TYPE_PRIMARY,
    description="Primary model profile for general chat and reasoning.",
    model_name=DEFAULT_TEXT_TO_TEXT_MODEL,  # = "qwen3-30b-a3b-q4-k-m"
    parameters=ModelParameters(
        num_ctx=40960,
        repeat_last_n=128,
        repeat_penalty=1.1,
        temperature=0.65,
        seed=-1,
        num_predict=-1,
        top_k=30,
        top_p=0.85,
        # ... additional parameters
    ),
    system_prompt="You are a helpful AI assistant. Provide clear, accurate, and helpful responses.",
)
```

### Model Parameters Explanation

- **`num_ctx`**: Context window size (40,960 tokens)
- **`temperature`**: Randomness control (0.65 = moderate creativity)
- **`top_p`**: Nucleus sampling threshold (0.85)
- **`top_k`**: Top-K sampling limit (30)
- **`repeat_penalty`**: Repetition reduction (1.1)

## Usage Patterns

### 1. Getting User's Primary Profile

```python
from utils.model_profile import get_model_profile
from models.model_profile_type import ModelProfileType

# Get user's configured primary profile
primary_profile = await get_model_profile(user_id, ModelProfileType.Primary)
```

### 2. Creating Custom Profiles

```python
from models.model_profile import ModelProfile
from models.model_parameters import ModelParameters

custom_profile = ModelProfile(
    user_id=user_id,
    name="My Custom Profile",
    model_name="qwen3-30b-a3b-q4-k-m",
    parameters=ModelParameters(
        temperature=0.8,
        num_ctx=100000,
        # ... custom parameters
    ),
    system_prompt="Custom instructions...",
    type=ModelProfileType.Primary
)

# Save to database
created_profile = await storage.model_profile.create_model_profile(custom_profile)
```

### 3. Updating User Configuration

```python
from db import storage

# Get user config
user_config = await storage.user_config.get_user_config(user_id)

# Update primary profile reference
user_config.model_profiles.primary_profile_id = custom_profile.id

# Save changes
await storage.user_config.update_user_config(user_id, user_config)
```

## API Endpoints

### Model Profiles

- **GET** `/models/profiles` - List user's profiles
- **GET** `/models/profiles/{id}` - Get specific profile
- **POST** `/models/profiles` - Create new profile
- **PUT** `/models/profiles/{id}` - Update profile
- **DELETE** `/models/profiles/{id}` - Delete profile

### Available Models

- **GET** `/models/` - List all available models from `.models.json`

## Testing

### Running Profile Tests

```bash
# Test with specific model
kubectl exec -it -n ollama $POD_NAME -- /app/v.sh composer python -m debug.test_composer_real_e2e

# Test model profile functionality
kubectl exec -it -n ollama $POD_NAME -- /app/v.sh server python -m debug.test_model_profiles
```

### Test Configuration

The test system supports multiple models and automatically creates test profiles:

```python
available_models = [
    "llama-chat-summary-3_2-3b-q5-k-m",
    "openai-gpt-oss-20b-uncensored-q5_1", 
    "qwen3-30b-a3b-q4-k-m",  # Primary default
    "qwen2.5-vl-32b-instruct-q4-k-m",
]
```

## Best Practices

### 1. Profile Design

- **Task-Specific**: Create profiles optimized for specific use cases
- **Parameter Tuning**: Adjust temperature/context based on task requirements
- **System Prompts**: Use clear, specific instructions for each profile type
- **Naming Convention**: Use descriptive names for custom profiles

### 2. Model Selection

- **Primary Tasks**: Use `qwen3-30b-a3b-q4-k-m` for general reasoning
- **Code Tasks**: Use `qwen3-coder-30b-a3b` for programming
- **Summarization**: Use `llama-chat-summary-3_2-3b-q5-k-m` for text summarization
- **Vision Tasks**: Use `qwen2.5-vl-32b-instruct-q4-k-m` for image analysis

### 3. Performance Optimization

- **Context Size**: Balance between capability and memory usage
- **Temperature**: Lower for factual tasks, higher for creative tasks
- **Caching**: Leverage profile caching for frequently used configurations

## Troubleshooting

### Common Issues

1. **Profile Not Found**: Check profile ID exists and belongs to user
2. **Model Unavailable**: Verify model exists in `.models.json`
3. **Parameter Errors**: Validate parameter ranges and types
4. **Permission Denied**: Ensure user owns the profile or it's a system default

### Debug Commands

```bash
# Check available models
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "from models.default_model_profiles import DEFAULT_PROFILES; print(list(DEFAULT_PROFILES.keys()))"

# Validate model configuration
kubectl exec -n ollama $POD_NAME -- /app/v.sh server python -c "from db import storage; import asyncio; print(asyncio.run(storage.model_profile.list_model_profiles_by_user('test_user')))"
```

## Implementation Details

### Database Schema

Profiles are stored in the `model_profiles` table with JSON serialization for:

- `parameters` (ModelParameters object)
- `circuit_breaker` (CircuitBreakerConfig object)
- `gpu_config` (GpuConfig object)

### Profile Types (Enum Values)

```python
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
```

### System UUIDs

Default profiles use predetermined UUIDs for consistency:

```python
DEFAULT_PRIMARY_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_SUMMARIZATION_PROFILE_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
# ... etc
```

## Conclusion

The model profile system provides a flexible, hierarchical configuration approach that balances ease of use with customization power. The **primary model profile is already configured to use `qwen3-30b-a3b-q4-k-m`** as requested, and the system supports seamless model switching and task-specific optimization.

All other default model profiles remain unchanged, providing a stable foundation for the various specialized tasks throughout the system.
