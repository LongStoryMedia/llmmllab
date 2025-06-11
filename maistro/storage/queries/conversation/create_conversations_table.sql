-- Create conversations table with TimescaleDB compatible schema
CREATE TABLE IF NOT EXISTS conversations(
  id serial,
  user_id text NOT NULL,
  title text DEFAULT 'New conversation',
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  PRIMARY KEY (id, created_at)
);

