-- Create the message_contents table as a hypertable without FK constraint
CREATE TABLE IF NOT EXISTS message_contents(
    id serial,
    message_id integer NOT NULL,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    type TEXT NOT NULL,
    text_content text,
    url text,
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable with optimal chunk interval
SELECT
    create_hypertable('message_contents', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '3 days');

-- Create a function to check referential integrity
CREATE OR REPLACE FUNCTION check_message_exists()
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
DROP TRIGGER IF EXISTS ensure_message_exists ON message_contents;

-- Create a trigger to enforce integrity
CREATE TRIGGER ensure_message_exists
    BEFORE INSERT OR UPDATE ON message_contents
    FOR EACH ROW
    EXECUTE FUNCTION check_message_exists();

-- Create additional indexes for message_contents
CREATE INDEX IF NOT EXISTS idx_message_contents_message_id ON message_contents(message_id);

CREATE INDEX IF NOT EXISTS idx_message_contents_message_time ON message_contents(message_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_message_contents_type ON message_contents(type);

-- Enable compression on message_contents hypertable
ALTER TABLE message_contents SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id');

-- Add data compression policy for message_contents
SELECT
    add_compression_policy('message_contents', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for message_contents data (365 days)
SELECT
    add_retention_policy('message_contents', INTERVAL '365 days', if_not_exists => TRUE);

