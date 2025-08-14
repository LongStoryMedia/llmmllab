-- Update conversation title 
UPDATE conversations 
SET title = $1, updated_at = NOW() 
WHERE id = $2