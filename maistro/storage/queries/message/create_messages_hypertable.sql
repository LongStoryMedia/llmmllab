-- Create hypertable for messages with optimal chunk interval
SELECT
  create_hypertable('messages', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '3 days')
