-- Get a model profile by its ID and user ID
SELECT
  id,
  user_id,
  name,
  description,
  model_name,
  parameters,
  system_prompt,
  created_at,
  updated_at,
  model_version,
  type,
  circuit_breaker,
  gpu_config
FROM
  model_profiles
WHERE
  id = $1
  AND user_id = $2;

