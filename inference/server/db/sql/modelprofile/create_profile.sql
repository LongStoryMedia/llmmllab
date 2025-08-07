-- Insert a new model profile or update if it already exists
INSERT INTO model_profiles(user_id, name, description, model_name, parameters, system_prompt, model_version, type)
  VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (id)
  DO UPDATE SET
    name = EXCLUDED.name,
    description = EXCLUDED.description,
    model_name = EXCLUDED.model_name,
    parameters = EXCLUDED.parameters,
    system_prompt = EXCLUDED.system_prompt,
    model_version = EXCLUDED.model_version,
    type = EXCLUDED.type,
    updated_at = NOW()
  RETURNING
    id;

