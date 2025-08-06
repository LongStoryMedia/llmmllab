-- Create a new conversation for a user
INSERT INTO conversations(user_id, title)
  VALUES ($1, $2)
RETURNING
  id
