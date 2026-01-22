-- Add a new message to a conversation
INSERT INTO messages(conversation_id, role)
  VALUES ($1, $2)
RETURNING
  id
