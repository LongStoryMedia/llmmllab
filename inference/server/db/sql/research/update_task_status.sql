-- Update the status and error message of a research task
UPDATE research_tasks
SET status = $2, error_message = $3, updated_at = NOW()
WHERE id = $1
RETURNING id, updated_at;
