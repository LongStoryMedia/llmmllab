-- Delete a dynamic tool
DELETE FROM dynamic_tools
WHERE id = $1
    AND user_id = $2;

