-- Add thought to database
INSERT INTO thoughts(message_id, text)
    VALUES ($1, $2)
RETURNING
    id, message_id, text, created_at;

