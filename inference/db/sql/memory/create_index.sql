-- Create indexes for memories table
CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);

CREATE INDEX IF NOT EXISTS idx_memories_vector ON memories USING hnsw(embedding vector_cosine_ops);

