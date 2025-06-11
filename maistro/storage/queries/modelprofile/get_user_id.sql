-- Get user ID for a model profile
SELECT
  user_id
FROM
  model_profiles
WHERE
  id = $1
