-- Get all messages for a conversation (simple query without aggregation, with pagination)
SELECT
    m.id,
    m.conversation_id,
    m.role,
    m.created_at
FROM
    messages m
WHERE
    m.conversation_id = $1
ORDER BY
    m.created_at ASC
LIMIT $2 OFFSET $3
