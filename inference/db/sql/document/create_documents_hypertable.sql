-- Convert documents table to TimescaleDB hypertable
-- This must be done after the basic table is created
SELECT
    create_hypertable('documents', 'created_at', chunk_time_interval => INTERVAL '1 month', if_not_exists => TRUE, migrate_data => TRUE);

-- Enable compression for older document chunks (1 week old)
-- This is especially beneficial for document content which can be large
ALTER TABLE documents SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id, user_id', timescaledb.compress_orderby = 'created_at DESC');

-- Set up automatic compression policy for documents older than 1 week
SELECT
    add_compression_policy('documents', INTERVAL '1 week', if_not_exists => TRUE);

