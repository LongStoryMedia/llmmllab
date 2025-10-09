-- Search dynamic tools by semantic similarity for a specific user (basic columns only)
-- $1: user_id
-- $2: query embedding vector
-- $3: similarity threshold (optional, default 0.7)
-- $4: maximum number of results (limit)
-- $5: offset for pagination
-- Returns tools ordered by similarity score
SELECT
    id,
    user_id,
    name,
    description,
    code,
    function_name,
    embedding,
    parameters,
    created_at,
    updated_at,
    1 -(embedding <=> $2) AS similarity_score -- Cosine distance converted to similarity
FROM
    dynamic_tools
WHERE
    user_id = $1
    AND embedding IS NOT NULL
    AND 1 -(embedding <=> $2) > $3 -- Filter by similarity threshold
ORDER BY
    similarity_score DESC -- Higher similarity first
LIMIT $4 OFFSET $5;

