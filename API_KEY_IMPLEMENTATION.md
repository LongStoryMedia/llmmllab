# API Key Authentication Implementation Summary

## Overview
Successfully implemented API key authentication alongside the existing OAuth2.0 JWT authentication. This enables IDE integration while maintaining strong security through API key management.

## Components Implemented

### 1. **Data Models** (`schemas/` and `inference/models/`)
- **`schemas/api_key.yaml`**: Defines the API key storage model with fields for:
  - Unique ID and user ownership
  - Hashed key (SHA-256, never plaintext)
  - Scopes (chat, generate, embed)
  - Creation, last used, and expiration timestamps
  - Revocation status
  
- **`schemas/api_key_response.yaml`**: Response model that includes the plaintext key (returned only on creation)

### 2. **Database Layer** (`inference/db/`)

#### Schema Files (`sql/api_key/`)
- `create_api_keys_table.sql`: Creates the api_keys table with indexes for efficient lookups
- `create_api_key.sql`: Insert query for creating new keys
- `get_api_key_by_hash.sql`: Query for validating keys during auth
- `list_api_keys_for_user.sql`: List user's keys
- `update_last_used.sql`: Track key usage
- `revoke_api_key.sql`: Soft delete keys
- `delete_api_key.sql`: Permanent deletion

#### Storage Service (`api_key_storage.py`)
- `ApiKeyStorage` class with methods:
  - `generate_key()`: Creates cryptographically secure random keys
  - `hash_key()`: SHA-256 hashing for secure storage
  - `create_api_key()`: Create and store new keys
  - `validate_api_key()`: Check key validity and expiration
  - `list_api_keys_for_user()`: User's keys
  - `revoke_api_key()`: Soft revocation
  - `delete_api_key()`: Permanent deletion
  - `update_last_used()`: Usage tracking

### 3. **Authentication Middleware** (`inference/server/middleware/auth.py`)

#### New `ApiKeyValidator` Class
- Lazy-loads API key storage on first use
- Validates API keys against database
- Updates last_used_at timestamp
- Returns `TokenValidationResult` with user_id and scopes

#### Enhanced `AuthMiddleware`
- Now supports both OAuth2 (Bearer tokens) and API keys
- Falls back from JWT to API key validation
- Supports two authentication headers:
  - `Authorization: Bearer <api_key>` (backward compatible)
  - `X-API-Key: <api_key>` (preferred for IDE integration)

### 4. **Ollama Router Updates** (`inference/server/routers/ollama.py`)

Implemented `verify_api_key_access()` dependency that:
- Validates API key from X-API-Key or Bearer token
- Returns authenticated user_id for subsequent use
- All Ollama endpoints now require API key authentication:
  - `/api/generate`
  - `/api/chat`
  - `/api/tags`
  - `/api/show`
  - `/api/create`
  - `/api/copy`
  - `/api/delete`
  - `/api/pull`
  - `/api/push`

### 5. **API Key Management Router** (`inference/server/routers/api_key.py`)

New endpoints for key management:

**`POST /api-keys/create`**
- Create new API key
- Request: `{ name, scopes[], expires_in_days }`
- Returns plaintext key (only once!)
- Response includes all metadata

**`GET /api-keys/list`**
- List all user's keys (without plaintext key)
- Shows name, creation date, expiration, scopes, revocation status

**`POST /api-keys/revoke`**
- Soft-revoke a key (can be re-enabled conceptually)
- Request: `{ key_id }`

**`POST /api-keys/delete`**
- Permanently delete a key
- Request: `{ key_id }`

**`GET /api-keys/info/{key_id}`**
- Get details about specific key
- Verifies ownership before returning

### 6. **Database Integration** (`inference/db/__init__.py`)

- Registered `ApiKeyStorage` in the `Storage` class
- Added initialization in the connection pool setup
- Automatically initialized on app startup

### 7. **Database Initialization** (`inference/db/init_db.py`)

- Added API key table initialization step (depends on users table)
- Runs during application startup with idempotent SQL

## Security Features

1. **Key Hashing**: Keys stored as SHA-256 hashes, never plaintext
2. **One-Time Display**: Plaintext key shown only on creation
3. **Expiration Support**: Optional time-limited keys
4. **Revocation**: Soft and hard deletion options
5. **Scope Limiting**: Keys can be restricted to specific API scopes
6. **Ownership Verification**: Keys only accessible by owner
7. **Usage Tracking**: Last used timestamp for auditing
8. **Cryptographic Generation**: Uses `secrets.token_hex()` for secure randomness

## Usage Examples

### Creating an API Key
```bash
curl -X POST http://localhost:8000/api-keys/create \
  -H "Authorization: Bearer <jwt_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "IDE Integration",
    "scopes": ["chat", "generate", "embed"],
    "expires_in_days": 90
  }'
```

Response:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user123",
  "key": "a1b2c3d4e5f6...",  // Store this securely!
  "name": "IDE Integration",
  "created_at": "2026-01-19T10:00:00Z",
  "expires_at": "2026-04-19T10:00:00Z",
  "scopes": ["chat", "generate", "embed"]
}
```

### Using API Key with Ollama Endpoint
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: a1b2c3d4e5f6..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Listing Your Keys
```bash
curl http://localhost:8000/api-keys/list \
  -H "Authorization: Bearer <jwt_token>"
```

## Authentication Priority

When a request comes in:
1. Checks for `Authorization: Bearer <token>`
   - First tries JWT validation (OAuth2)
   - Falls back to API key validation
2. Checks for `X-API-Key: <key>` header
   - Validates against database
3. Returns 401 if no valid auth found

## Benefits for IDE Integration

1. **No Browser Auth**: IDEs can use API keys instead of OAuth flow
2. **Long-Lived Access**: Keys can expire or be manually revoked
3. **Scope Control**: Limit key permissions to specific operations
4. **Easy Revocation**: Instantly disable compromised keys
5. **Usage Tracking**: See when keys were last used
6. **Multiple Keys**: Create separate keys for different IDEs/machines

## Files Created/Modified

**Created:**
- `schemas/api_key.yaml`
- `schemas/api_key_response.yaml`
- `inference/models/api_key.py`
- `inference/models/api_key_response.py`
- `inference/db/sql/api_key/*.sql` (7 files)
- `inference/db/api_key_storage.py`
- `inference/server/routers/api_key.py`

**Modified:**
- `inference/db/__init__.py` - Added ApiKeyStorage registration
- `inference/db/init_db.py` - Added API key table initialization
- `inference/server/middleware/auth.py` - Added ApiKeyValidator and dual auth
- `inference/server/routers/ollama.py` - Updated all endpoints to require API key
- `inference/server/app.py` - Imported and registered api_key router
- `inference/models/__init__.py` - Exported new models

## Next Steps (Optional)

1. **Key Rotation**: Implement key rotation for enhanced security
2. **Rate Limiting**: Add per-key rate limits
3. **Audit Logging**: Log all API key operations
4. **Admin Dashboard**: UI for viewing/managing keys
5. **Key Templates**: Pre-defined scope templates for common use cases
6. **Device Binding**: Optional IP address restrictions per key

## Testing

To validate the implementation:

```bash
# 1. Create a test API key
KEY_RESPONSE=$(curl -X POST http://localhost:8000/api-keys/create \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{"name":"test","scopes":["chat"]}')

API_KEY=$(echo $KEY_RESPONSE | jq -r '.key')

# 2. Use it with Ollama endpoint
curl -X POST http://localhost:8000/api/tags \
  -H "X-API-Key: $API_KEY"

# 3. List keys
curl http://localhost:8000/api-keys/list \
  -H "Authorization: Bearer <jwt>"

# 4. Revoke key
curl -X POST http://localhost:8000/api-keys/revoke \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d "{\"key_id\":\"<id>\"}"
```

All tasks completed! The API key system is now fully integrated with your authentication middleware and ready for IDE integration.
