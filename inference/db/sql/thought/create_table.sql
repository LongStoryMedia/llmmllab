-- Create thoughts table as a hypertable following TimescaleDB pattern
-- One-to-many relationship with messages (one message can have many thoughts)
CREATE TABLE IF NOT EXISTS thoughts(
    id serial,
    message_id integer NOT NULL,
    text text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable with optimal chunk interval
SELECT
    create_hypertable('thoughts', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '3 days');

-- Create a function to check that the referenced message exists
CREATE OR REPLACE FUNCTION check_message_exists_for_thought()
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
DROP TRIGGER IF EXISTS ensure_message_exists_thought ON thoughts;

-- Create trigger to enforce referential integrity
CREATE TRIGGER ensure_message_exists_thought
    BEFORE INSERT OR UPDATE ON thoughts
    FOR EACH ROW
    EXECUTE FUNCTION check_message_exists_for_thought();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_thoughts_message_id ON thoughts(message_id);

CREATE INDEX IF NOT EXISTS idx_thoughts_message_time ON thoughts(message_id, created_at DESC);

-- Enable compression on thoughts hypertable
ALTER TABLE thoughts SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id');

-- Add compression policy for thoughts
SELECT
    add_compression_policy('thoughts', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for thoughts data (365 days)
SELECT
    add_retention_policy('thoughts', INTERVAL '365 days', if_not_exists => TRUE);

