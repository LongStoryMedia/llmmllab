-- Create a new dynamic tool with LangChain BaseTool interface support
INSERT INTO dynamic_tools(user_id, name, description, code, function_name, embedding, args_schema, return_direct, tags, metadata, handle_tool_error, handle_validation_error, response_format, parameters)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
RETURNING
    id, user_id, name, description, code, function_name, embedding, args_schema, return_direct, tags, metadata, handle_tool_error, handle_validation_error, response_format, parameters, created_at, updated_at;

