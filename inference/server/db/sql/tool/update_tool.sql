-- Update an existing dynamic tool
UPDATE
    dynamic_tools
SET
    name = $3,
    description = $4,
    code = $5,
    function_name = $6,
    embedding = $7,
    parameters = $8
WHERE
    id = $1
    AND user_id = $2
RETURNING
    id,
    user_id,
    name,
    description,
    code,
    function_name,
    embedding,
    parameters,
    created_at,
    updated_at;

