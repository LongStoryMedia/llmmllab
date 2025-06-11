-- Add data compression policy for messages
SELECT
  add_compression_policy('messages', INTERVAL '7 days', if_not_exists => TRUE);

