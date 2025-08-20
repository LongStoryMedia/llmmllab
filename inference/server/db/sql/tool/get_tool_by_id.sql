-- Get a dynamic tool by ID for a specific user
SELECT id, user_id, name, description, code, function_name, embedding, parameters, created_at, updated_at
FROM dynamic_tools
WHERE id = $1 AND user_id = $2;
