-- Get a single conversation by ID
SELECT
    id,
    user_id,
    title,
    created_at,
    updated_at
FROM
    conversations
WHERE
    id = $1
