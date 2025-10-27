-- LangGraph AsyncPostgresSaver checkpoint tables
-- These tables store workflow state for multi-turn conversations
-- Based on LangGraph's AsyncPostgresSaver schema

-- Main checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

-- Checkpoint writes table for atomic operations
CREATE TABLE IF NOT EXISTS checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    channel TEXT NOT NULL,
    type TEXT,
    value JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id ON checkpoints(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at ON checkpoints(created_at);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_thread_id ON checkpoint_writes(thread_id);
CREATE INDEX IF NOT EXISTS idx_checkpoint_writes_checkpoint_id ON checkpoint_writes(checkpoint_id);

-- Add comments for documentation
COMMENT ON TABLE checkpoints IS 'LangGraph workflow checkpoints for persistent state across conversation turns';
COMMENT ON TABLE checkpoint_writes IS 'Atomic checkpoint write operations for LangGraph state updates';
COMMENT ON COLUMN checkpoints.thread_id IS 'Conversation ID used as thread identifier';
COMMENT ON COLUMN checkpoints.checkpoint IS 'Serialized workflow state including todos and planning context';
COMMENT ON COLUMN checkpoints.metadata IS 'Additional metadata about the checkpoint';