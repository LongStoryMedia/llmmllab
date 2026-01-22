-- Get thoughts by message ID
SELECT
    id,
    message_id,
    text,
    created_at
FROM
    thoughts
WHERE
    message_id = $1
ORDER BY
    created_at ASC;

