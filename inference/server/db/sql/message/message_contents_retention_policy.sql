-- Add retention policy for message_contents
SELECT
    add_retention_policy('message_contents', INTERVAL '365 days', if_not_exists => TRUE);

