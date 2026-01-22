# API Key Authentication - Quick Reference

## For IDE Integration

### Step 1: Create an API Key
Use your OAuth2 JWT token to create an API key:

```bash
curl -X POST http://localhost:8000/api-keys/create \
  -H "Authorization: Bearer <your_jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My IDE",
    "scopes": ["chat", "generate", "embed"],
    "expires_in_days": 365
  }'
```

Save the returned `key` value - you won't see it again!

### Step 2: Configure Your IDE
In your IDE settings, add:
```
API_KEY=<the_key_from_step_1>
INFERENCE_SERVER=http://localhost:8000
```

### Step 3: Make Requests
Use either header format:

**Preferred (X-API-Key):**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","messages":[...]}'
```

**Alternative (Bearer token):**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer <api_key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama2","messages":[...]}'
```

## Key Management

### List Your Keys
```bash
curl http://localhost:8000/api-keys/list \
  -H "Authorization: Bearer <jwt_token>"
```

### Check Key Status
```bash
curl http://localhost:8000/api-keys/info/{key_id} \
  -H "Authorization: Bearer <jwt_token>"
```

### Revoke a Key (Disable)
```bash
curl -X POST http://localhost:8000/api-keys/revoke \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"key_id":"<key_id>"}'
```

### Delete a Key (Permanent)
```bash
curl -X POST http://localhost:8000/api-keys/delete \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{"key_id":"<key_id>"}'
```

## Available Endpoints

All Ollama-compatible endpoints now support API key auth:
- `POST /api/chat` - Chat completions
- `POST /api/generate` - Text generation
- `GET /api/tags` - List models
- `POST /api/show` - Model info
- `POST /api/embed` - Embeddings
- (Other management endpoints...)

## Authentication Priority

1. **Authorization: Bearer** header (tries JWT, then API key)
2. **X-API-Key** header (API key only)
3. Returns 401 if neither valid

## Scopes

When creating a key, you can restrict it to specific scopes:
- `chat` - Chat completion endpoints
- `generate` - Text generation
- `embed` - Embedding endpoints

Default: all scopes

## Best Practices

✅ **DO:**
- Store API keys in environment variables
- Create separate keys for different IDEs/machines
- Use expiring keys (set expires_in_days)
- Revoke keys immediately if compromised
- Check last_used_at to detect unused keys

❌ **DON'T:**
- Commit API keys to version control
- Share API keys across projects
- Use forever-keys for sensitive applications
- Log API keys in debug output
- Use root scopes if you only need chat

## Troubleshooting

**"Invalid or expired API key"**
- Check the key is correct and not revoked
- Check if it's expired (expires_at)
- Recreate if lost

**"API key required"**
- Add `X-API-Key` header or `Authorization: Bearer` header
- Make sure it's formatted correctly

**Lost my key**
- Can't recover it (never stored plaintext)
- Create a new one and delete the old one

## Python Example

```python
import requests

API_KEY = "your_api_key_here"
BASE_URL = "http://localhost:8000"

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(
    f"{BASE_URL}/api/chat",
    headers=headers,
    json={
        "model": "llama2",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False
    }
)

print(response.json())
```

## Node.js Example

```javascript
const API_KEY = "your_api_key_here";
const BASE_URL = "http://localhost:8000";

fetch(`${BASE_URL}/api/chat`, {
  method: "POST",
  headers: {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    model: "llama2",
    messages: [{role: "user", content: "Hello"}],
    stream: false
  })
})
.then(res => res.json())
.then(data => console.log(data))
.catch(err => console.error(err));
```

## VS Code Extension Setup

```json
{
  "inferenceApi": {
    "baseUrl": "http://localhost:8000",
    "apiKey": "${env:INFERENCE_API_KEY}",
    "model": "llama2"
  }
}
```

Then set environment variable:
```bash
export INFERENCE_API_KEY="your_api_key_here"
```
