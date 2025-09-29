-- Count dynamic tools by semantic similarity (for pagination)
-- $1: query embedding vector
-- $2: similarity threshold (optional, default 0.7)
SELECT 
    COUNT(*) as total_count
FROM 
    dynamic_tools
WHERE 
    embedding IS NOT NULL
    AND 1 - (embedding <=> $1) > $2;  -- Filter by similarity threshold
