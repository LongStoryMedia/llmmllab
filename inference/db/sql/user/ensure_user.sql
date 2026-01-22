-- Create a user if they don't exist already
INSERT INTO users(id)
  VALUES ($1)
ON CONFLICT (id)
  DO NOTHING
