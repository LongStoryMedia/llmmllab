-- Add retention policy for messages data (365 days)
SELECT
  add_retention_policy('messages', INTERVAL '365 days', if_not_exists => TRUE);

