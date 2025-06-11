-- Get a model profile by its ID
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
  id = $1;

