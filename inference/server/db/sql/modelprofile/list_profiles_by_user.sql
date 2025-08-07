-- List all model profiles for a specific user
SELECT
  id,
  user_id,
  name,
  description,
  model_name,
  parameters,
  system_prompt,
  model_version,
  type,
  created_at,
  updated_at
FROM
  model_profiles
WHERE
  user_id = $1
ORDER BY
  updated_at DESC
