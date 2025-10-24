-- Create analyses table as a hypertable following TimescaleDB pattern
-- One-to-many relationship with messages (one message can have many analyses)
-- Schema matches IntentAnalysis model
CREATE TABLE IF NOT EXISTS analyses(
    id serial,
    message_id integer NOT NULL,
    workflow_type text NOT NULL,
    complexity_level text NOT NULL,
    required_capabilities jsonb NOT NULL,
    domain_specificity numeric(3, 2) NOT NULL CHECK (domain_specificity >= 0.0 AND domain_specificity <= 1.0),
    reusability_potential numeric(3, 2) NOT NULL CHECK (reusability_potential >= 0.0 AND reusability_potential <= 1.0),
    confidence numeric(3, 2) NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    response_format text,
    technical_domain text,
    requires_tools boolean NOT NULL DEFAULT FALSE,
    requires_custom_tools boolean NOT NULL DEFAULT FALSE,
    tool_complexity_score numeric(3, 2) NOT NULL CHECK (tool_complexity_score >= 0.0 AND tool_complexity_score <= 1.0),
    computational_requirements jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to hypertable with optimal chunk interval
SELECT
    create_hypertable('analyses', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '3 days');

-- Create a function to check that the referenced message exists
CREATE OR REPLACE FUNCTION check_message_exists_for_analysis()
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
DROP TRIGGER IF EXISTS ensure_message_exists_analysis ON analyses;

-- Create trigger to enforce referential integrity
CREATE TRIGGER ensure_message_exists_analysis
    BEFORE INSERT OR UPDATE ON analyses
    FOR EACH ROW
    EXECUTE FUNCTION check_message_exists_for_analysis();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_analyses_message_id ON analyses(message_id);

CREATE INDEX IF NOT EXISTS idx_analyses_message_time ON analyses(message_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_analyses_workflow_type ON analyses(workflow_type);

CREATE INDEX IF NOT EXISTS idx_analyses_complexity_level ON analyses(complexity_level);

CREATE INDEX IF NOT EXISTS idx_analyses_technical_domain ON analyses(technical_domain)
WHERE
    technical_domain IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_analyses_requires_tools ON analyses(requires_tools);

CREATE INDEX IF NOT EXISTS idx_analyses_requires_custom_tools ON analyses(requires_custom_tools);

CREATE INDEX IF NOT EXISTS idx_analyses_confidence ON analyses(confidence);

CREATE INDEX IF NOT EXISTS idx_analyses_required_capabilities ON analyses USING GIN(required_capabilities);

CREATE INDEX IF NOT EXISTS idx_analyses_computational_requirements ON analyses USING GIN(computational_requirements);

-- Enable compression on analyses hypertable
ALTER TABLE analyses SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id');

-- Add compression policy for analyses
SELECT
    add_compression_policy('analyses', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for analyses data (365 days)
SELECT
    add_retention_policy('analyses', INTERVAL '365 days', if_not_exists => TRUE);

