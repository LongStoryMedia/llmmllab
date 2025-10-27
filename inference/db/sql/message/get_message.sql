-- Get a message by ID (simple query without aggregation)
SELECT
    m.id,
    m.conversation_id,
    m.role,
    m.created_at
FROM
    messages m
WHERE
    m.id = $1
