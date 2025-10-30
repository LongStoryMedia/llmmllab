-- Create tool_calls table as a hypertable following TimescaleDB pattern
-- One-to-many relationship with messages (one message can have many tool calls)
-- Schema matches ToolExecutionResult model
CREATE TABLE IF NOT EXISTS tool_calls(
    id serial,
    message_id integer NOT NULL,
    tool_name text NOT NULL,
    execution_id text,
    success boolean NOT NULL,
    args jsonb,
    result_data jsonb,
    error_message text,
    execution_time_ms numeric(10, 3) CHECK (execution_time_ms >= 0),
    resource_usage jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable with optimal chunk interval
SELECT
    create_hypertable('tool_calls', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '3 days');

-- Create a function to check that the referenced message exists
CREATE OR REPLACE FUNCTION check_message_exists_for_tool_call()
    RETURNS TRIGGER
    AS $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            messages
        WHERE
            id = NEW.message_id) THEN
    RAISE EXCEPTION 'Referenced message does not exist';
END IF;
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

-- Drop the trigger if it exists to avoid errors on re-runs
DROP TRIGGER IF EXISTS ensure_message_exists_tool_call ON tool_calls;

-- Create trigger to enforce referential integrity
CREATE TRIGGER ensure_message_exists_tool_call
    BEFORE INSERT OR UPDATE ON tool_calls
    FOR EACH ROW
    EXECUTE FUNCTION check_message_exists_for_tool_call();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tool_calls_message_id ON tool_calls(message_id);

CREATE INDEX IF NOT EXISTS idx_tool_calls_message_time ON tool_calls(message_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);

CREATE INDEX IF NOT EXISTS idx_tool_calls_success ON tool_calls(success);

CREATE INDEX IF NOT EXISTS idx_tool_calls_execution_id ON tool_calls(execution_id)
WHERE
    execution_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_calls_args ON tool_calls USING GIN(args)
WHERE
    args IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_calls_result_data ON tool_calls USING GIN(result_data)
WHERE
    result_data IS NOT NULL;

-- Enable compression on tool_calls hypertable
-- Use created_at for ordering (time dimension) and message_id for segmenting (different columns)
ALTER TABLE tool_calls SET (
    timescaledb.compress, 
    timescaledb.compress_orderby = 'created_at DESC',
    timescaledb.compress_segmentby = 'message_id'
);

-- Add compression policy for tool_calls
SELECT
    add_compression_policy('tool_calls', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for tool_calls data (365 days)
SELECT
    add_retention_policy('tool_calls', INTERVAL '365 days', if_not_exists => TRUE);

