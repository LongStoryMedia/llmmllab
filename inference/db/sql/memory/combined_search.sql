-- Combined vector similarity and text search for messages
SELECT
  m.id,
  m.conversation_id,
  m.role,
  m.content,
  m.created_at,
(0.7 *(1 -(e.embedding <=> $1)) + 0.3 * ts_rank_cd(to_tsvector('english', m.content), websearch_to_tsquery('english', $2))) AS combined_score
FROM
  memories e
  JOIN messages m ON e.source_id = m.id
WHERE
  to_tsvector('english', m.content) @@ websearch_to_tsquery('english', $2)
WHERE
  e.source = 'message'
ORDER BY
  combined_score DESC
LIMIT $3
