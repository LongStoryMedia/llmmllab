-- Get all messages for a conversation with content, ordered chronologically
-- Exclude messages that have been summarized (source_ids is JSONB array)
-- JOIN with message_contents to get actual content data
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
    AND m.id NOT IN (
        SELECT
            CAST(jsonb_array_elements_text(source_ids) AS INTEGER)
        FROM
            summaries
        WHERE
            conversation_id = $1
            AND level = 1)
ORDER BY
    m.created_at ASC
