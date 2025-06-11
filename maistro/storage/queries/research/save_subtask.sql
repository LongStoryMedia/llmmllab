-- Save a research subtask with upsert behavior
INSERT INTO research_subtasks(task_id, question_id, status, created_at, updated_at)
  VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (task_id, question_id)
  DO UPDATE SET
    status = EXCLUDED.status,
    updated_at = EXCLUDED.updated_at
  RETURNING
    id
