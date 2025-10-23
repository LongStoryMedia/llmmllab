-- Get tool calls by message ID
SELECT id, message_id, tool_data, created_at
FROM tool_calls
WHERE message_id = $1
ORDER BY created_at ASC;