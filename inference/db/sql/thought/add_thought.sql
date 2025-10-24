-- Add thought to database
INSERT INTO thoughts(message_id, text, created_at)
    VALUES ($1, $2, COALESCE($3, CURRENT_TIMESTAMP))
RETURNING
    id, message_id, text, created_at;

