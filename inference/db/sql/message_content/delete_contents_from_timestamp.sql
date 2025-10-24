-- Delete message contents for messages created at or after a specific timestamp
-- This leverages TimescaleDB's time-series optimization for efficient bulk deletion
DELETE FROM message_contents 
WHERE message_id IN (
    SELECT id FROM messages 
    WHERE conversation_id = $1 
    AND created_at >= $2
);