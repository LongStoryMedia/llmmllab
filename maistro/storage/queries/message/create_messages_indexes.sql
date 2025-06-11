-- Add unique index on messages.id and created_at for FK references (required by TimescaleDB)
CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_id_createdat_unique ON messages(id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);

CREATE INDEX IF NOT EXISTS idx_messages_id_role ON messages(id, ROLE);

-- Create full-text search index on messages content
CREATE INDEX IF NOT EXISTS idx_messages_content_fts ON messages USING GIN(to_tsvector('english', content));

