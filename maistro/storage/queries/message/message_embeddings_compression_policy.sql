-- Add compression policy for message embeddings
SELECT
  add_compression_policy('message_embeddings', INTERVAL '7 days', if_not_exists => TRUE);

