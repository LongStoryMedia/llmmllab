-- Create vector similarity search index
CREATE INDEX IF NOT EXISTS idx_message_embeddings_embedding ON message_embeddings USING HNSW(embedding vector_cosine_ops)
