-- Get all messages for a conversation with all contents aggregated as JSON, ordered chronologically
SELECT
    m.id,
    m.conversation_id,
    m.role,
    m.created_at,
    COALESCE(JSON_AGG(JSON_BUILD_OBJECT('type', mc.type, 'text_content', mc.text_content, 'url', mc.url, 'created_at', mc.created_at)
        ORDER BY mc.id) FILTER (WHERE mc.message_id IS NOT NULL), '[]'::json) AS contents
FROM
    messages m
    LEFT JOIN message_contents mc ON m.id = mc.message_id
WHERE
    m.conversation_id = $1
GROUP BY
    m.id,
    m.conversation_id,
    m.role,
    m.created_at
ORDER BY
    m.created_at ASC
LIMIT $2 OFFSET $3
