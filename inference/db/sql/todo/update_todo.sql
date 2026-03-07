-- Update a todo item
UPDATE todos 
SET 
    title = $3,
    description = $4,
    status = $5,
    priority = $6,
    due_date = $7,
    updated_at = NOW()
WHERE id = $1 AND user_id = $2
RETURNING id, user_id, conversation_id, title, description, status, priority, due_date, created_at, updated_at;