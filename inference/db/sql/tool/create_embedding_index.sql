-- Create a vector similarity index on the embedding field
-- This will improve performance of semantic search queries
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_embedding_cosine 
ON dynamic_tools 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
