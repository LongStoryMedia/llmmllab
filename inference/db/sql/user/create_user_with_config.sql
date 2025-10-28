-- Create a user with default configuration if they don't exist already
INSERT INTO users(id, config)
  VALUES ($1, $2)
ON CONFLICT (id)
  DO UPDATE SET 
    config = COALESCE(users.config, EXCLUDED.config)