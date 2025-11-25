-- Update message role (typically role doesn't change, but included for completeness)
UPDATE messages 
SET role = $2
WHERE id = $1;