-- Add retention policy for memories data (365 days)
SELECT
  add_retention_policy('memories', INTERVAL '365 days', if_not_exists => TRUE);

