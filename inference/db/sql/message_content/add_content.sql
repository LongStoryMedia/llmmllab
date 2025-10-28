-- Add a new message content (simple insert without conflict resolution)
INSERT INTO message_contents(message_id, type, text_content, url)
    VALUES ($1, $2, $3, $4)
    RETURNING
        id;

