-- Delete an API key
DELETE FROM api_keys
WHERE id = $1 AND user_id = $2
RETURNING id;
