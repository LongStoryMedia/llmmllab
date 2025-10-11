-- Update an existing dynamic tool with LangChain BaseTool interface support
UPDATE
    dynamic_tools
SET
    name = $3,
    description = $4,
    code = $5,
    function_name = $6,
    embedding = $7,
    args_schema = $8,
    return_direct = $9,
    tags = $10,
    metadata = $11,
    handle_tool_error = $12,
    handle_validation_error = $13,
    response_format = $14,
    parameters = $15
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
    args_schema,
    return_direct,
    tags,
    metadata,
    handle_tool_error,
    handle_validation_error,
    response_format,
    parameters,
    created_at,
    updated_at;

