-- Create analyses table as a hypertable following TimescaleDB pattern
-- One-to-many relationship with messages (one message can have many analyses)

CREATE TABLE IF NOT EXISTS analyses (
    id SERIAL,
    message_id INTEGER NOT NULL,
    analysis_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable with optimal chunk interval
SELECT create_hypertable('analyses', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '3 days');

-- Create a function to check that the referenced message exists
CREATE OR REPLACE FUNCTION check_message_exists_for_analysis()
    RETURNS TRIGGER
    AS $$
BEGIN
    IF NOT EXISTS(
        SELECT 1
        FROM messages
        WHERE id = NEW.message_id
    ) THEN
        RAISE EXCEPTION 'Referenced message does not exist';
    END IF;
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

-- Drop the trigger if it exists to avoid errors on re-runs
DROP TRIGGER IF EXISTS ensure_message_exists_analysis ON analyses;

-- Create trigger to enforce referential integrity
CREATE TRIGGER ensure_message_exists_analysis
    BEFORE INSERT OR UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION check_message_exists_for_analysis();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analyses_message_id ON analyses(message_id);
CREATE INDEX IF NOT EXISTS idx_analyses_message_time ON analyses(message_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_data ON analyses USING GIN (analysis_data);

-- Enable compression on analyses hypertable
ALTER TABLE analyses SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id');

-- Add compression policy for analyses
SELECT add_compression_policy('analyses', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for analyses data (365 days)
SELECT add_retention_policy('analyses', INTERVAL '365 days', if_not_exists => TRUE);