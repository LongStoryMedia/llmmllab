-- Insert a new model profile or update if it already exists
INSERT INTO model_profiles(id, user_id, name, description, model_name, parameters, system_prompt, model_version, type, circuit_breaker, created_at, updated_at)
  VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW())
