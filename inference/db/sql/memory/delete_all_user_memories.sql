-- Delete all memories for a specific user
DELETE FROM memories
WHERE user_id = $1;

