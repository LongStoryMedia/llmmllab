-- create unique index on id and created_at for FK references (required by TimescaleDB)
CREATE UNIQUE INDEX IF NOT EXISTS idx_summaries_id_createdat_unique ON summaries(id, created_at);

-- Add unique index on summaries.id and created_at for FK references (required by TimescaleDB)
CREATE INDEX IF NOT EXISTS idx_summaries_conversation_id_createdat ON summaries(conversation_id, created_at);

-- Create additional indexes for summaries
CREATE INDEX IF NOT EXISTS idx_summaries_conversation_id ON summaries(conversation_id);

CREATE INDEX IF NOT EXISTS idx_summaries_conversation_level ON summaries(conversation_id, level);

