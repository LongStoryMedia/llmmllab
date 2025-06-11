-- Update an existing model profile
UPDATE
  model_profiles
SET
  name = $2,
  description = $3,
  model_name = $4,
  parameters = $5,
  system_prompt = $6,
  model_version = $7,
  type = $8,
  updated_at = NOW()
WHERE
  id = $1
RETURNING
  updated_at
