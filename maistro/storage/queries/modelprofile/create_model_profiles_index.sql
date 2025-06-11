-- Create index on model_profiles for user_id
CREATE INDEX IF NOT EXISTS idx_model_profiles_user_id ON model_profiles(user_id);

