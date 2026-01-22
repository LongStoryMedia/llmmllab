-- List all API keys for a user
SELECT id, user_id, key_hash, name, created_at, last_used_at, expires_at, is_revoked, scopes
FROM api_keys
WHERE user_id = $1
ORDER BY created_at DESC;
