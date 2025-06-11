-- Create messages table with TimescaleDB compatible schema
CREATE TABLE IF NOT EXISTS messages(
  id serial,
  conversation_id integer NOT NULL,
  role TEXT NOT NULL,
  content text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  PRIMARY KEY (id, created_at))
