-- $1 user_id (non optional)
-- $2 conversation_id (optional)
-- $3 limit (optional, default 25)
-- $4 offset (optional, default 0)
SELECT
    id,
    filename,
    thumbnail,
    format,
    width,
    height,
    conversation_id,
    user_id,
    created_at,
    COUNT(*) OVER () AS total_count
FROM
    images
WHERE
    user_id = $1
    AND ($2::bigint IS NULL
        OR conversation_id = $2::bigint)
ORDER BY
    created_at DESC
LIMIT COALESCE($3, 25) OFFSET COALESCE($4, 0)
