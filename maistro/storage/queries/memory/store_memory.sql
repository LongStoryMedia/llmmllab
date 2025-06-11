-- Store a memory with its embedding
INSERT INTO memories(user_id, source_id, source, embedding, role)
  VALUES ($1, $2, $3, $4, $5)
