-- Create users table
CREATE TABLE IF NOT EXISTS users(
  id text PRIMARY KEY,
  username text,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  config jsonb DEFAULT NULL)
