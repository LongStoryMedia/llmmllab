-- Get todos by conversation_id for a user
SELECT
    id,
    user_id,
    conversation_id,
    title,
    description,
    status,
    priority,
    due_date,
    created_at,
    updated_at
FROM
    todos
WHERE
    user_id = $1
    AND conversation_id = $2
ORDER BY
    created_at DESC;

