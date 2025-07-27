-- Get all messages for a conversation, ordered chronologically
SELECT
    id,
    conversation_id,
    ROLE,
    created_at
FROM
    messages
WHERE
    conversation_id = $1
ORDER BY
    created_at ASC
