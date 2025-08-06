-- Add data compression policy for memories
SELECT
  add_compression_policy('memories', INTERVAL '30 days', if_not_exists => TRUE);

