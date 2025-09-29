SELECT
    create_hypertable('search_topic_syntheses', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => interval '3 days');

-- Enable compression on search_topic_syntheses hypertable
ALTER TABLE search_topic_syntheses SET (timescaledb.compress, timescaledb.compress_segmentby = 'id');

-- Add data compression policy for search_topic_syntheses
SELECT
    add_compression_policy('search_topic_syntheses', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for search_topic_syntheses data (365 days)
SELECT
    add_retention_policy('search_topic_syntheses', INTERVAL '365 days', if_not_exists => TRUE);

