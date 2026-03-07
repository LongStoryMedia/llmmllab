-- Revoke an API key
UPDATE api_keys
SET is_revoked = TRUE
WHERE id = $1 AND user_id = $2
RETURNING id;
