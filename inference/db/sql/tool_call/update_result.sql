-- Update tool call result data
UPDATE
    tool_calls
SET
    result_data = $3,
    success = $4,
    error_message = $5,
    execution_time_ms = $6
WHERE
    message_id = $1
    AND execution_id = $2
RETURNING
    id,
    message_id,
    tool_name AS name,
    execution_id,
    success,
    args,
    result_data,
    error_message,
    execution_time_ms,
    resource_usage,
    created_at;

