-- Insert a new model profile (ID auto-generated for user profiles)
INSERT INTO model_profiles(user_id, name, description, model_name, parameters, system_prompt, model_version, type)
  VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
RETURNING
  id;

