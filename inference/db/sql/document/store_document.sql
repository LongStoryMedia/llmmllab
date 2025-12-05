-- Store a new document
INSERT INTO documents(message_id, user_id, filename, content_type, file_size, content, text_content)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING
    id, created_at;

