-- Get API key by hash (for authentication)
SELECT id, user_id, key_hash, name, created_at, last_used_at, expires_at, is_revoked, scopes
FROM api_keys
WHERE key_hash = $1 AND NOT is_revoked
  AND (expires_at IS NULL OR expires_at > NOW());
