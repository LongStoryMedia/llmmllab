-- Add retention policy for summaries data (365 days)
SELECT
  add_retention_policy('summaries', INTERVAL '365 days', if_not_exists => TRUE);

