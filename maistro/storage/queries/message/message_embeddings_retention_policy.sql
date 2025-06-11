-- Add retention policy for message embeddings
SELECT
  add_retention_policy('message_embeddings', INTERVAL '90 days', if_not_exists => TRUE);

