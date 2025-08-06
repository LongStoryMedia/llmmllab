-- Create research subtasks table
CREATE TABLE IF NOT EXISTS research_subtasks(
  id serial PRIMARY KEY,
  task_id integer NOT NULL,
  title text NOT NULL,
  description text,
  status text NOT NULL DEFAULT 'pending',
  result text,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  FOREIGN KEY (task_id) REFERENCES research_tasks(id) ON DELETE CASCADE
);

