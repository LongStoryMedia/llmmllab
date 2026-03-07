-- Get document by ID
SELECT 
    id,
    conversation_id,
    user_id,
    filename,
    content_type,
    file_size,
    content,
    text_content,
    created_at,
    updated_at
FROM documents 
WHERE id = $1;