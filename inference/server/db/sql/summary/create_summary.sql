INSERT INTO summaries(conversation_id, content, level, source_ids)
  VALUES ($1, $2, $3, $4)
RETURNING
  id;

