-- Full text search in messages content within a specific conversation
SELECT
  m.id,
  m.conversation_id,
  m.role,
  m.content,
  m.created_at,
  ts_rank_cd(to_tsvector('english', m.content), websearch_to_tsquery('english', $1)) AS rank
FROM
  messages m
WHERE
  m.conversation_id = $3
  AND to_tsvector('english', m.content) @@ websearch_to_tsquery('english', $1)
ORDER BY
  rank DESC
LIMIT $2
