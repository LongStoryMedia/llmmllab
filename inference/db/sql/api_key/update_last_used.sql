-- Update last_used_at timestamp
UPDATE api_keys
SET last_used_at = NOW()
WHERE id = $1 AND NOT is_revoked;
