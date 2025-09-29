-- Create memory indexes for efficient search
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_id_createdat_unique ON memories(id, created_at);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);

CREATE INDEX idx_memories_source_id_source ON memories(source_id, source);

-- Create vector similarity search index on memories
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING HNSW(embedding vector_cosine_ops);

SET max_parallel_workers_per_gather = 4;

