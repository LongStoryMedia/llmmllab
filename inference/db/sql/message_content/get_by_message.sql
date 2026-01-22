-- Get message contents by message ID
SELECT
    mc.id,
    mc.message_id,
    mc.type,
    mc.text_content AS text,
    mc.url,
    mc.format,
    mc.name,
    mc.created_at
FROM
    message_contents mc
WHERE
    mc.message_id = $1
ORDER BY
    mc.id
