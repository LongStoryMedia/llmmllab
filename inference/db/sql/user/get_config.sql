-- Get user configuration from the users table
SELECT
  config
FROM
  users
WHERE
  id = $1
