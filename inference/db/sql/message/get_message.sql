-- Get a message by ID with content
SELECT
    id,
    conversation_id,
    role,
    created_at,
FROM
    messages 
WHERE
    m.id = $1
