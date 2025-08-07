-- Create hypertable for conversations with optimal chunk interval
SELECT
  create_hypertable('conversations', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '7 days')
