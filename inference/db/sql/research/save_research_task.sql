-- Save a research subtask with upsert behavior
INSERT INTO research_tasks(user_id, conversation_id, query)
  VALUES ($1, $2, $3)
RETURNING
  id
