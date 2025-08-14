-- Create messages table with TimescaleDB compatible schema
CREATE TABLE IF NOT EXISTS messages(
  id serial,
  conversation_id integer NOT NULL,
  role TEXT NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  PRIMARY KEY (id, created_at)
);

-- Create hypertable for messages with optimal chunk interval
SELECT
  create_hypertable('messages', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => interval '3 days');

-- Create additional indexes for messages
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_time ON messages(conversation_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_id_createdat_unique ON messages(id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_id_role ON messages(id, ROLE);

-- Enable compression on messages hypertable
ALTER TABLE messages SET (timescaledb.compress, timescaledb.compress_segmentby = 'conversation_id');

-- Add data compression policy for messages
SELECT
  add_compression_policy('messages', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for messages data (365 days)
SELECT
  add_retention_policy('messages', INTERVAL '365 days', if_not_exists => TRUE);

