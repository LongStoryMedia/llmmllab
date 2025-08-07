-- Create additional indexes for conversations
CREATE INDEX IF NOT EXISTS idx_conversations_user_time ON conversations(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);

