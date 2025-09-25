-- Get a message by ID (content is fetched separately)
SELECT
    m.id,
    m.conversation_id,
    m.ROLE,
    m.created_at
FROM
    messages m
WHERE
    m.id = $1
