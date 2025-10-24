-- Add tool call to database using ToolExecutionResult schema
INSERT INTO tool_calls(message_id, tool_name, execution_id, success, args, result_data, error_message, execution_time_ms, resource_usage, created_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, COALESCE($10, NOW()))
RETURNING
    id;

