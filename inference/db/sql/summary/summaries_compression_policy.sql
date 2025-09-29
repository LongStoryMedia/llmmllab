-- Add data compression policy for summaries
SELECT
  add_compression_policy('summaries', INTERVAL '14 days', if_not_exists => TRUE);

