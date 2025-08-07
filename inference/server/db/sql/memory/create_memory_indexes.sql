-- Create memory indexes for efficient search
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_id_createdat_unique ON memories(id, created_at);

-- Add unique index on memories.id and created_at for FK references (required by TimescaleDB)
CREATE INDEX IF NOT EXISTS idx_memories_conversation_id_createdat ON memories(conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);

CREATE INDEX idx_memories_source_id_source ON memories(source_id, source);

-- Create vector similarity search index on memories
CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories USING HNSW(embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_memories_conversation_id ON memories(conversation_id);

SET max_parallel_workers_per_gather = 4;

