INSERT INTO message_contents(message_id, type, text_content, url)
    VALUES ($1, $2, $3, $4);

-- Return the ID of the newly created message content
