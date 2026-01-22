-- Add unique constraint on conversations.id to allow foreign key references
-- This is needed because the table has a composite primary key (id, created_at)
-- but documents table needs to reference just the id column
ALTER TABLE conversations
    ADD CONSTRAINT conversations_id_unique UNIQUE (id);

