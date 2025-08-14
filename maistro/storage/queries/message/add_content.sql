INSERT INTO message_contents(message_id, message_created_at, created_at, type, text_content, url)
    VALUES ($1, NOW(), NOW(), $2, $3, $4);

-- Return the ID of the newly created message content
