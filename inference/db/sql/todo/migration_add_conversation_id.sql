-- Migration: Add conversation_id column to todos table
-- This migration adds conversation linking capability to todos
-- Add the new column (nullable since existing todos won't have a conversation)
ALTER TABLE todos
    ADD COLUMN IF NOT EXISTS conversation_id integer;

-- Create index for efficient conversation-based queries
CREATE INDEX IF NOT EXISTS idx_todos_conversation ON todos(user_id, conversation_id)
WHERE
    conversation_id IS NOT NULL;

-- Optional: Add foreign key constraint if conversations table exists
-- ALTER TABLE todos
-- ADD CONSTRAINT fk_todos_conversation
-- FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL;
