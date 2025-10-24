-- Create todos table as a hypertable following TimescaleDB pattern
-- User-owned todo items with priority and status tracking
CREATE TABLE IF NOT EXISTS todos(
    id serial,
    user_id text NOT NULL,
    conversation_id integer,
    title text NOT NULL,
    description text,
    status text NOT NULL CHECK (status IN ('not-started', 'in-progress', 'completed', 'cancelled')),
    priority text NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    due_date timestamptz,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable with optimal chunk interval
SELECT
    create_hypertable('todos', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '1 week');

-- Index for efficient user queries
CREATE INDEX IF NOT EXISTS idx_todos_user_id ON todos(user_id, created_at DESC);

-- Index for conversation-based queries
CREATE INDEX IF NOT EXISTS idx_todos_conversation ON todos(conversation_id, created_at DESC) WHERE conversation_id IS NOT NULL;

-- Index for status filtering
CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status, created_at DESC);

-- Index for priority filtering
CREATE INDEX IF NOT EXISTS idx_todos_priority ON todos(priority, created_at DESC);

-- Index for due date queries
CREATE INDEX IF NOT EXISTS idx_todos_due_date ON todos(due_date)
WHERE
    due_date IS NOT NULL;

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_todo_updated_at()
    RETURNS TRIGGER
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

-- Trigger to automatically update updated_at on row changes
DROP TRIGGER IF EXISTS trigger_update_todo_updated_at ON todos;

CREATE TRIGGER trigger_update_todo_updated_at
    BEFORE UPDATE ON todos
    FOR EACH ROW
    EXECUTE FUNCTION update_todo_updated_at();

