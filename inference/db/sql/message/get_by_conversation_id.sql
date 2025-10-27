-- Get all messages for a conversation with all contents, tool_calls, and thoughts aggregated as JSON, ordered chronologically
SELECT
    m.id,
    m.conversation_id,
    m.role,
    m.created_at,
    COALESCE((
        SELECT
            JSON_AGG(JSON_BUILD_OBJECT('type', mc.type, 'text_content', mc.text_content, 'url', mc.url, 'created_at', mc.created_at)
        ORDER BY mc.id)
        FROM message_contents mc
        WHERE
            mc.message_id = m.id), '[]'::json) AS contents,
    COALESCE((
        SELECT
            JSON_AGG(JSON_BUILD_OBJECT('id', tc.id, 'tool_name', tc.tool_name, 'execution_id', tc.execution_id, 'success', tc.success, 'args', tc.args, 'result_data', tc.result_data, 'error_message', tc.error_message, 'execution_time_ms', tc.execution_time_ms, 'resource_usage', tc.resource_usage, 'created_at', tc.created_at)
            ORDER BY tc.created_at)
        FROM tool_calls tc
        WHERE
            tc.message_id = m.id), '[]'::json) AS tool_calls,
    COALESCE((
        SELECT
            JSON_AGG(JSON_BUILD_OBJECT('id', th.id, 'message_id', th.message_id, 'text', th.text, 'created_at', th.created_at)
            ORDER BY th.created_at)
        FROM thoughts th
        WHERE
            th.message_id = m.id), '[]'::json) AS thoughts
FROM
    messages m
WHERE
    m.conversation_id = $1
ORDER BY
    m.created_at ASC
LIMIT $2 OFFSET $3
