-- Search dynamic tools by semantic similarity
-- $1: query embedding vector
-- $2: similarity threshold (optional, default 0.7)
-- $3: maximum number of results (limit)
-- $4: offset for pagination
-- Returns tools ordered by similarity score
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
    updated_at,
    1 -(embedding <=> $1) AS similarity_score -- Cosine distance converted to similarity
FROM
    dynamic_tools
WHERE
    embedding IS NOT NULL
    AND 1 -(embedding <=> $1) > $2 -- Filter by similarity threshold
ORDER BY
    similarity_score DESC -- Higher similarity first
LIMIT $3 OFFSET $4;

