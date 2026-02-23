 # OpenAI API Routers

This directory contains auto-generated FastAPI routers for all OpenAI API endpoints extracted from `openai.documented.yml`.

## Overview

- **23 root path routers** covering all major OpenAI API resources
- Each router file corresponds to a root path (e.g., `chat.py`, `audio.py`, `models.py`)
- All endpoints currently raise `NotImplementedError` and are ready for implementation
- Request/response types are automatically imported from generated Pydantic models in `inference/models/openai/`

## Router Files

| Router | Paths | Example |
|--------|-------|---------|
| `chat.py` | `/chat/completions`, `/chat/completions/{id}`, `/chat/completions/{id}/messages` | POST `/v1/chat/completions` |
| `audio.py` | `/audio/speech`, `/audio/transcriptions`, `/audio/translations`, `/audio/voices`, `/audio/voice_consents` | POST `/v1/audio/transcriptions` |
| `models.py` | `/models`, `/models/{model}` | GET `/v1/models` |
| `embeddings.py` | `/embeddings` | POST `/v1/embeddings` |
| `images.py` | `/images/generations`, `/images/edits`, `/images/variations` | POST `/v1/images/generations` |
| `assistants.py` | `/assistants`, `/assistants/{id}` | GET `/v1/assistants` |
| `threads.py` | `/threads`, `/threads/{id}`, `/threads/{id}/runs`, `/threads/{id}/messages` | POST `/v1/threads` |
| `files.py` | `/files`, `/files/{id}`, `/files/{id}/content` | GET `/v1/files` |
| `fine_tuning.py` | `/fine_tuning/jobs`, `/fine_tuning/jobs/{id}`, etc. | GET `/v1/fine_tuning/jobs` |
| `batches.py` | `/batches`, `/batches/{id}`, `/batches/{id}/cancel` | POST `/v1/batches` |
| `completions.py` | `/completions` | POST `/v1/completions` (legacy) |
| `uploads.py` | `/uploads`, `/uploads/{id}`, `/uploads/{id}/parts`, `/uploads/{id}/complete` | POST `/v1/uploads` |
| `vector_stores.py` | `/vector_stores`, `/vector_stores/{id}`, `/vector_stores/{id}/files` | GET `/v1/vector_stores` |
| `moderations.py` | `/moderations` | POST `/v1/moderations` |
| `evals.py` | `/evals`, `/evals/{id}`, `/evals/{id}/runs`, etc. | GET `/v1/evals` |
| `organization.py` | `/organization/*` resources | Various |
| `projects.py` | `/projects/*` resources | Various |
| `conversations.py` | `/conversations/{id}/items` | GET `/v1/conversations/{id}/items` |
| `containers.py` | `/containers`, `/containers/{id}`, `/containers/{id}/files` | GET `/v1/containers` |
| `responses.py` | `/responses` (Responses API) | POST `/v1/responses` |
| `realtime.py` | Realtime API endpoints | Various |
| `chatkit.py` | Chat Kit specific endpoints | Various |
| `videos.py` | Video generation endpoints | POST `/v1/videos/generations` |

## Implementing Endpoints

Each endpoint is scaffolded with:
- Proper HTTP method (GET, POST, PUT, DELETE, etc.)
- Correct path with parameter placeholders (e.g., `{completion_id}`)
- Pydantic request/response models already imported
- Operation ID from the OpenAPI spec as the function name

### Example implementation for `chat.py`

```python
@router.post("/completions")
async def createChatCompletion(body: CreateChatCompletionRequest) -> CreateChatCompletionResponse:
    """Create chat completion"""
    # TODO: Implement actual logic
    user_id = get_user_id()  # Get from auth context
    
    # Process the request
    response = CreateChatCompletionResponse(
        id=f"chatcmpl-{uuid4().hex[:12]}",
        object="chat.completion",
        created=int(datetime.now().timestamp()),
        model=body.model,
        choices=[...],
        usage=...,
    )
    return response
```

### Example implementation for endpoints with path parameters

```python
@router.get("/completions/{completion_id}")
async def getChatCompletion(completion_id: str) -> CreateChatCompletionResponse:
    """Retrieve stored chat completion"""
    # Retrieve from database
    completion = await storage.get_completion(completion_id)
    
    if not completion:
        raise HTTPException(status_code=404, detail="Completion not found")
    
    return completion
```

## Available Pydantic Models

All response types are available from `inference/models/openai/`:

```python
# Import specific models
from inference.models.openai.create_chat_completion_response import CreateChatCompletionResponse
from inference.models.openai.create_speech_request import CreateSpeechRequest
from inference.models.openai.list_models_response import ListModelsResponse

# Or use generic dict when model is not available
from typing import Dict, Any
response: Dict[str, Any] = {...}
```

To see all available models:
```bash
ls inference/models/openai/ | grep -E '\.py$' | wc -l
```

## Code Generation

These routers were generated from the OpenAPI spec using:

```bash
python generate_routers.py
```

The script:
1. Parses `openai.documented.yml`
2. Groups endpoints by root path
3. Extracts request/response schemas
4. Generates FastAPI router files
5. Creates import files for easy integration

### Regenerating routers

If you update `openai.documented.yml`:

```bash
# Remove old routers
rm -rf inference/server/routers/openai

# Regenerate
python generate_routers.py
```

## Notes

- All endpoints currently raise `NotImplementedError`
- Request body parameters are passed as `body: ModelType` parameter
- Query and path parameters are passed as individual parameters
- Response types are annotated with their Pydantic models
- Import paths use snake_case for model file names (e.g., `CreateChatCompletionResponse` is in `create_chat_completion_response.py`)

## Related Files

- `../../openai.documented.yml` - OpenAPI spec source
- `../../generate_routers.py` - Router generation script
- `../../inference/models/openai/` - Generated Pydantic models
- `../../inference/server/routers/openai.py` - Existing OpenAI compatibility router
