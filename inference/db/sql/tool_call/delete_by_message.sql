-- Delete tool calls by message ID
DELETE FROM tool_calls
WHERE message_id = $1;

