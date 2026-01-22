-- Add a new message content (simple insert without conflict resolution)
INSERT INTO message_contents(message_id, type, text_content, url, format, name, created_at)
    VALUES ($1, $2, $3, $4, $5, $6, COALESCE($7, NOW()))
RETURNING
    id;

