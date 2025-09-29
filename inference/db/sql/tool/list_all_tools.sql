-- List all dynamic tools with pagination
SELECT id, user_id, name, description, code, function_name, embedding, parameters, created_at, updated_at
FROM dynamic_tools
ORDER BY created_at DESC
LIMIT $1  -- Limit number of results
OFFSET $2;  -- Skip first N results for pagination
