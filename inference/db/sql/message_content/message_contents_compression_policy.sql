-- Add compression policy for message_contents
SELECT
    add_compression_policy('message_contents', INTERVAL '7 days', if_not_exists => TRUE);

