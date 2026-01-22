-- Get all conversations for a user, ordered by recency
SELECT id, user_id, title, created_at, updated_at
FROM conversations
WHERE user_id = $1
ORDER BY updated_at DESC