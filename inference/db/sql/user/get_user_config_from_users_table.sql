-- Get user configuration directly from the users table
SELECT config FROM users WHERE id = $1