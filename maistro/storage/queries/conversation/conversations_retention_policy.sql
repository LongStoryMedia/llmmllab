-- Add retention policy for conversations data (365 days)
SELECT
  add_retention_policy('conversations', INTERVAL '365 days', if_not_exists => TRUE);

