-- Get a specific todo by ID and user_id
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
    id = $1
    AND user_id = $2;

