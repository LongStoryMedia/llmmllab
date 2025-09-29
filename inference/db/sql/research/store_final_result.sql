-- Store the final research result as a JSON array
UPDATE
  research_tasks
SET
  results = jsonb_build_array($2),
  updated_at = NOW()
WHERE
  id = $1
RETURNING
  updated_at
