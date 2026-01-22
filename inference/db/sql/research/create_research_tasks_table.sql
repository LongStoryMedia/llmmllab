-- Create research tasks table
CREATE TABLE IF NOT EXISTS research_tasks(
  id serial PRIMARY KEY,
  user_id text NOT NULL,
  conversation_id text NOT NULL,
  query text NOT NULL,
  status text NOT NULL DEFAULT 'pending',
  result text,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW()
);

