-- List all model profiles for a specific user
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
  user_id = $1
ORDER BY
  created_at DESC
