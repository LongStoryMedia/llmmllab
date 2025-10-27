-- Get all messages for a conversation (simple query without aggregation)
-- Exclude messages that have been summarized (source_ids is JSONB array)
SELECT
    m.id,
    m.conversation_id,
    m.role,
    m.created_at
FROM
    messages m
WHERE
    m.conversation_id = $1
    AND m.id NOT IN (
        SELECT
            CAST(jsonb_array_elements_text(source_ids) AS integer)
        FROM
            summaries
        WHERE
            conversation_id = $1
            AND level = 1)
ORDER BY
    m.created_at ASC
