-- Migration to add conversation_id column to todos table
-- This allows todos to be associated with specific conversations
-- Add conversation_id column
ALTER TABLE todos
    ADD COLUMN IF NOT EXISTS conversation_id integer;

-- Create index for conversation-based queries
CREATE INDEX IF NOT EXISTS idx_todos_conversation ON todos(conversation_id, created_at DESC)
WHERE
    conversation_id IS NOT NULL;

