# API Key Authentication Architecture

## System Overview

```
┌─────────────┐
│   IDE       │
│  Extension  │
└──────┬──────┘
       │ X-API-Key or Bearer
       │
       ▼
┌──────────────────────────────────┐
│   API Request with Auth Header   │
└──────────────────────┬───────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │  AuthMiddleware              │
        │  ├─ Checks Authorization     │
        │  ├─ Tries JWT validation     │
        │  └─ Falls back to API key    │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
    ┌────────────┐          ┌─────────────────┐
    │ JWTValidator          │ ApiKeyValidator │
    │ (OAuth2)   │          │ (API Key Auth)  │
    └────────────┘          └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │ ApiKeyStorage       │
                            │ ├─ validate_api_key │
                            │ ├─ hash_key         │
                            │ └─ generate_key     │
                            └────────┬────────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │  Database (PostgreSQL)
                            │  ├─ api_keys table  │
                            │  └─ indexes         │
                            └─────────────────────┘
```

## Data Flow: Authentication

### API Key Creation
```
User Request (OAuth2 JWT)
    │
    ▼
POST /api-keys/create
    │
    ▼
ApiKeyRouter.create_api_key()
    │
    ├─ Validate JWT → get user_id
    │
    ├─ ApiKeyStorage.generate_key()
    │   └─ Creates: secrets.token_hex(32)
    │
    ├─ ApiKeyStorage.hash_key()
    │   └─ Creates: SHA-256(key)
    │
    ├─ INSERT INTO api_keys (user_id, key_hash, name, scopes, expires_at)
    │
    └─ Return ApiKeyResponse with plaintext key
       (Plaintext key shown ONLY ONCE)
```

### API Key Validation
```
IDE/Client Request
    │
    ├─ Header: X-API-Key: <key>  OR
    └─ Header: Authorization: Bearer <key>
       │
       ▼
    AuthMiddleware.authenticate()
       │
       ├─ Extract key from header
       │
       ├─ ApiKeyValidator.validate_api_key(<key>)
       │   │
       │   ├─ hash_key(<key>) → key_hash
       │   │
       │   ├─ Query: SELECT * FROM api_keys
       │   │          WHERE key_hash = $1
       │   │          AND NOT is_revoked
       │   │          AND (expires_at IS NULL OR expires_at > NOW())
       │   │
       │   ├─ If found:
       │   │   ├─ update_last_used(key_id)
       │   │   └─ Return TokenValidationResult(user_id, scopes)
       │   │
       │   └─ If not found: Return None
       │
       ├─ Store in request.state.auth:
       │   ├─ USER_ID
       │   ├─ TOKEN_CLAIMS (includes scopes)
       │   ├─ IS_ADMIN (false for API keys)
       │   └─ REQUEST_ID
       │
       └─ Proceed to endpoint handler
```

## Database Schema

```sql
CREATE TABLE api_keys (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  key_hash text NOT NULL UNIQUE,           -- SHA-256 hash
  name text NOT NULL,                       -- User-friendly name
  created_at timestamptz NOT NULL DEFAULT NOW(),
  last_used_at timestamptz,                -- For audit trail
  expires_at timestamptz,                  -- NULL = never expires
  is_revoked boolean NOT NULL DEFAULT FALSE,
  scopes text[] NOT NULL DEFAULT ARRAY[]   -- e.g., ["chat", "generate"]
);

-- Indexes for efficient lookups
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash) 
  WHERE NOT is_revoked;  -- Only non-revoked keys
CREATE INDEX idx_api_keys_expires_at ON api_keys(expires_at) 
  WHERE expires_at IS NOT NULL AND NOT is_revoked;
```

## Component Architecture

### ApiKeyValidator (middleware/auth.py)

```python
class ApiKeyValidator:
    async def validate_api_key(api_key: str) -> Optional[TokenValidationResult]:
        # 1. Get storage service (lazy-load)
        # 2. Call storage.validate_api_key(api_key)
        #    - Hashes the key
        #    - Queries database
        #    - Checks expiration and revocation
        # 3. If valid:
        #    - Update last_used timestamp (fire-and-forget)
        #    - Return TokenValidationResult with scopes
        # 4. If invalid: Return None
```

### ApiKeyStorage (db/api_key_storage.py)

Core methods:
- `generate_key()` - Creates secure random key
- `hash_key(key)` - SHA-256 hashing
- `create_api_key(user_id, name, scopes, expires_in_days)` - Creates key
- `validate_api_key(key)` - Validates during auth
- `get_api_key_by_hash(key_hash)` - Database lookup
- `list_api_keys_for_user(user_id)` - List user's keys
- `revoke_api_key(key_id, user_id)` - Soft revocation
- `delete_api_key(key_id, user_id)` - Permanent deletion
- `update_last_used(key_id)` - Audit timestamp

### AuthMiddleware (middleware/auth.py)

Enhanced to support dual authentication:

```python
async def authenticate(request: Request):
    auth_header = request.headers.get("Authorization")
    api_key = request.headers.get("X-API-Key")
    
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # Try JWT first
        try:
            result = await self.validator.validate_token(token)
            return result
        except HTTPException:
            # Fall back to API key validation
            result = await self.api_key_validator.validate_api_key(token)
            if result:
                return result
            raise
    
    if api_key:
        result = await self.api_key_validator.validate_api_key(api_key)
        if result:
            return result
        raise HTTPException(401, "Invalid API key")
    
    raise HTTPException(401, "No auth header")
```

## Security Considerations

### Key Generation
- Uses `secrets.token_hex(32)` (cryptographically secure)
- 64-character hexadecimal string
- Entropy: 256 bits

### Key Storage
- Only SHA-256 hash stored in database
- Plaintext key shown only on creation
- Never transmitted back after creation

### Key Validation Flow
```
1. Client sends plaintext key in header
2. Server hashes it: SHA-256(key)
3. Query database with hash
4. Compare hash (never compare plaintext)
5. Check expiration and revocation
6. Update last_used timestamp
```

### Revocation Strategy
- `is_revoked` boolean flag (soft delete)
- Query filters: `WHERE NOT is_revoked`
- Can be extended to add revocation reason
- Audit trail available via `last_used_at`

### Expiration Handling
```sql
WHERE (expires_at IS NULL OR expires_at > NOW())
```
- NULL = never expires
- Expired keys automatically invalid
- Database time used (server-side validation)

## Integration Points

### Ollama Endpoints
Each endpoint uses `Depends(verify_api_key_access)`:
```python
@router.post("/api/chat")
async def chat(
    body: OllamaChatRequest,
    request: Request,
    user_id: str = Depends(verify_api_key_access)
):
    # user_id extracted from validated API key
```

### Management Router
- GET /api-keys/list
- POST /api-keys/create
- POST /api-keys/revoke
- POST /api-keys/delete
- GET /api-keys/info/{id}

All require OAuth2 authentication (JWT only, not API key)

## Usage Patterns

### For IDEs
```
1. User generates API key (once)
2. IDE stores key in config/env var
3. IDE uses key for all requests
4. Key validated on each request
5. Last used timestamp updated
```

### For Automation
```
1. Admin creates key with limited scopes
2. Script uses key for automated tasks
3. Key expires after N days
4. Admin rotates key periodically
5. Audit trail available for compliance
```

### For Development
```
1. Dev creates personal API key
2. Shares key only with team
3. Revokes when dev leaves
4. Creates new key for new dev
5. No need to change passwords
```

## Performance Characteristics

### Key Validation
- **Lookup**: Index on `key_hash` → O(1)
- **Expiration Check**: Simple timestamp comparison → O(1)
- **Revocation Check**: Boolean field → O(1)
- **Database Query**: Fully indexed, ~1-5ms typical

### Indexes
```sql
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash) 
  WHERE NOT is_revoked;
```
- Partial index: only non-revoked keys
- Reduces index size
- Speeds up valid key lookups

### Async Design
- Non-blocking database queries
- Fire-and-forget `update_last_used()`
- Connection pooling for efficiency

## Future Enhancements

### Phase 2
- IP address binding per key
- Rate limiting per key
- Scope enforcement in endpoints
- Key rotation mechanism
- Admin dashboard

### Phase 3
- Multi-factor authentication for key creation
- Hardware security key support
- OAuth2 token exchange
- API key webhooks
- Detailed audit logging

### Phase 4
- Machine learning for anomaly detection
- Automatic key rotation policies
- Zero-trust architecture
- Key escrow for compliance
- Decentralized key validation

## Testing Strategy

### Unit Tests
- Key generation (randomness)
- Key hashing (deterministic)
- Validation logic
- Expiration checks
- Revocation checks

### Integration Tests
- End-to-end key creation
- API authentication flow
- Database persistence
- Concurrent access
- Edge cases (expired, revoked)

### Security Tests
- Brute force resistance
- Key collision probability
- Timing attacks
- SQL injection prevention
- Authorization boundary checks
