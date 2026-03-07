-- Add a new todo item
INSERT INTO todos(user_id, conversation_id, title, description, status, priority, due_date)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING
    id, created_at, updated_at;

