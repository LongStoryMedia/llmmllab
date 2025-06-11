-- Create message_embeddings table for vector embeddings with fixed 768 dimensions
CREATE TABLE IF NOT EXISTS message_embeddings(
  message_id integer NOT NULL,
  embedding VECTOR(768),
  chunk_index integer NOT NULL DEFAULT 0,
  total_chunks integer NOT NULL DEFAULT 1,
  original_dimension integer NOT NULL DEFAULT 768,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  PRIMARY KEY (message_id, chunk_index, created_at))
