-- Get todos by status for a user
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
    AND status = $2
ORDER BY
    CASE priority
    WHEN 'urgent' THEN
        1
    WHEN 'high' THEN
        2
    WHEN 'medium' THEN
        3
    WHEN 'low' THEN
        4
    END,
    created_at DESC;

