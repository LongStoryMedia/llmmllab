-- Add a new message to a conversation
INSERT INTO messages(conversation_id, role, content)
  VALUES ($1, $2, $3)
RETURNING
  id
