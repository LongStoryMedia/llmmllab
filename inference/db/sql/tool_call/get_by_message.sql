-- Get tool calls by message ID with ToolExecutionResult schema
SELECT
    id,
    message_id,
    tool_name,
    execution_id,
    success,
    args,
    result_data,
    error_message,
    execution_time_ms,
    resource_usage,
    created_at
FROM
    tool_calls
WHERE
    message_id = $1
ORDER BY
    created_at ASC;

