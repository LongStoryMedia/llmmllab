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
    verbose = $10,
    tags = $11,
    metadata = $12,
    handle_tool_error = $13,
    handle_validation_error = $14,
    response_format = $15,
    parameters = $16
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
    verbose,
    tags,
    metadata,
    handle_tool_error,
    handle_validation_error,
    response_format,
    parameters,
    created_at,
    updated_at;

