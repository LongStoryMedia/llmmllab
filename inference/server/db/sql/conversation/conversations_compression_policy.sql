-- Add data compression policy for conversations
SELECT
  add_compression_policy('conversations', INTERVAL '30 days', if_not_exists => TRUE);

