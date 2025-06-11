-- Create hypertable for message_embeddings with optimal chunk interval
SELECT
  create_hypertable('message_embeddings', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '7 days')
