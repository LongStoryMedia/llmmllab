-- Create summaries table with TimescaleDB compatible schema
CREATE TABLE IF NOT EXISTS summaries(
  id serial,
  conversation_id integer NOT NULL,
  content text NOT NULL,
  level integer NOT NULL,
  source_ids jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  PRIMARY KEY (id, created_at))
