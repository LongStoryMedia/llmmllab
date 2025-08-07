-- Initialize memory schema for vector search
-- Create memories table for persistent memory storage-- Create memories table for storing user memories
CREATE TABLE IF NOT EXISTS memories(
  id serial,
  user_id text NOT NULL,
  source_id integer NOT NULL,
  source text NOT NULL,
  role text,
  embedding vector(768) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
  PRIMARY KEY (id, created_at)
);

-- index on source ID
CREATE INDEX IF NOT EXISTS idx_memories_source_id ON memories(source_id);

-- Create hypertable for memories with optimal chunk interval
SELECT
  create_hypertable('memories', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => interval '30 days');

