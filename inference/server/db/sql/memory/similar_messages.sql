-- Find semantically similar messages based on embedding distance
SELECT
  m.id,
  m.conversation_id,
  m.role,
  m.content,
  m.created_at,
  1 -(e.embedding <=> $1) AS similarity
FROM
  memories e
  JOIN messages m ON e.source_id = m.id
WHERE
  1 -(e.embedding <=> $1) > $2
  AND e.source = $4
ORDER BY
  similarity DESC
LIMIT $3
