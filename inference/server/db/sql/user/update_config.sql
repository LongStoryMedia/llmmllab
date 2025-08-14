-- Update user configuration in the users table
UPDATE
  users
SET
  config = $1
WHERE
  id = $2
