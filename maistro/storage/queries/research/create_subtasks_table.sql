-- Create the research subtasks table
CREATE TABLE IF NOT EXISTS research_subtasks(
  id serial PRIMARY KEY,
  task_id text NOT NULL REFERENCES research_tasks(id) ON DELETE CASCADE,
  question_id integer NOT NULL,
  status text NOT NULL,
  gathered_info jsonb,
  information_sources jsonb,
  synthesized_answer text,
  error_message text,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW(),
  UNIQUE (task_id, question_id))
