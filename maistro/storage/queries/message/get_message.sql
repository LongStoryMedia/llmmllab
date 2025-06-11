-- Get a message by ID
SELECT
    id,
    conversation_id,
    ROLE,
    content,
    created_at
FROM
    messages
WHERE
    id = $1
