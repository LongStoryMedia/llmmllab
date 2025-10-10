-- List all dynamic tools for a specific user with pagination
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
       user_id = $1
ORDER BY
       created_at DESC
LIMIT $2 -- Limit number of results
OFFSET $3;

-- Skip first N results for pagination
