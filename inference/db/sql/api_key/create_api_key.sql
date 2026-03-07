-- Create a new API key
INSERT INTO api_keys(user_id, key_hash, name, scopes, expires_at)
    VALUES ($1, $2, $3, $4, $5)
RETURNING
    id, user_id, key_hash, name, created_at, expires_at, scopes, is_revoked;

