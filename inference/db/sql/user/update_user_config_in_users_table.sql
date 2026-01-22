-- Update user configuration in the users table
UPDATE users SET config = $2 WHERE id = $1