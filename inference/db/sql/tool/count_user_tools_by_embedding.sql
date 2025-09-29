-- Count dynamic tools by semantic similarity for a specific user (for pagination)
-- $1: user_id
-- $2: query embedding vector
-- $3: similarity threshold (optional, default 0.7)
SELECT 
    COUNT(*) as total_count
FROM 
    dynamic_tools
WHERE 
    user_id = $1
    AND embedding IS NOT NULL
    AND 1 - (embedding <=> $2) > $3;  -- Filter by similarity threshold
