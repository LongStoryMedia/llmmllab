-- Get all documents for a conversation by joining through messages
SELECT
    d.id,
    d.message_id,
    d.user_id,
    d.filename,
    d.content_type,
    d.file_size,
    d.content,
    d.text_content,
    d.created_at,
    d.updated_at
FROM
    documents d
    INNER JOIN messages m ON d.message_id = m.id
WHERE
    m.conversation_id = $1
ORDER BY
    d.created_at ASC;

