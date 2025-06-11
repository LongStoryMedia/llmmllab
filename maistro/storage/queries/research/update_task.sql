-- Update a research task with completed_at field when applicable
UPDATE
  research_tasks
SET
  status = $2,
  error_message = $3,
  updated_at = NOW(),
  completed_at = CASE WHEN $2 IN ('COMPLETED', 'FAILED', 'CANCELED') THEN
    NOW()
  ELSE
    completed_at
  END
WHERE
  id = $1
RETURNING
  updated_at
