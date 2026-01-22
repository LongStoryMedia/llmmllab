-- Delete messages created at or after a specific timestamp
-- This leverages TimescaleDB's time-series optimization for efficient bulk deletion
DELETE FROM messages
WHERE conversation_id = $1
    AND created_at > $2;

