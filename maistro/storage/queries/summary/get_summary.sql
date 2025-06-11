SELECT
  id,
  conversation_id,
  content,
  level,
  source_ids,
  created_at
FROM
  summaries
WHERE
  id = $1
