-- Update status for a research subtask
UPDATE
  research_subtasks
SET
  status = $2,
  error_message = $3,
  updated_at = NOW()
WHERE
  task_id = $1
  AND question_id = $4
RETURNING
  id,
  updated_at
