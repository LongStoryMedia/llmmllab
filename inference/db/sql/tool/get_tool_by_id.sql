-- Get a dynamic tool by ID for a specific user
SELECT
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
       updated_at
FROM
       dynamic_tools
WHERE
       id = $1
       AND user_id = $2;

