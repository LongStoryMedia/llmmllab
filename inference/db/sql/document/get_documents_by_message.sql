-- Get all documents for a message
SELECT 
    id,
    message_id,
    user_id,
    filename,
    content_type,
    file_size,
    content,
    text_content,
    created_at,
    updated_at
FROM documents 
WHERE message_id = $1
ORDER BY created_at ASC;