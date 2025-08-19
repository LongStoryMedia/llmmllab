SELECT
    create_hypertable('search_topic_synthesis', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => interval '3 days');

-- Enable compression on search_topic_synthesis hypertable
ALTER TABLE search_topic_synthesis SET (timescaledb.compress, timescaledb.compress_segmentby = 'id');

-- Add data compression policy for search_topic_synthesis
SELECT
    add_compression_policy('search_topic_synthesis', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy for search_topic_synthesis data (365 days)
SELECT
    add_retention_policy('search_topic_synthesis', INTERVAL '365 days', if_not_exists => TRUE);

