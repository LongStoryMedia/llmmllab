-- Get all messages for a conversation with content, ordered chronologically
SELECT
    m.id,
    m.conversation_id,
    m.ROLE,
    m.created_at,
    mc.text_content as content,
    mc.type as content_type,
    mc.url as content_url
FROM
    messages m
LEFT JOIN 
    message_contents mc ON m.id = mc.message_id
WHERE
    m.conversation_id = $1
ORDER BY
    m.created_at ASC
LIMIT $2 OFFSET $3
