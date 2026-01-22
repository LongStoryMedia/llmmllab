-- Create hypertable for message_contents
SELECT
    create_hypertable('message_contents', 'created_at', if_not_exists => TRUE, chunk_time_interval => interval '3 days');

