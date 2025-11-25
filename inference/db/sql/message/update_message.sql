-- Update message role (typically role doesn't change, but included for completeness)
UPDATE
    messages
SET
    ROLE = $2
WHERE
    id = $1;

