-- Delete a todo item
DELETE FROM todos
WHERE id = $1
    AND user_id = $2
RETURNING
    id;

