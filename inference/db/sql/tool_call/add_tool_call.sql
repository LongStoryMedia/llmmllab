-- Add tool call to database
INSERT INTO tool_calls (
    message_id, 
    tool_data, 
    created_at
)
VALUES ($1, $2, COALESCE($3, NOW()))
RETURNING id;