-- Get a message by ID with content
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
    m.id = $1
