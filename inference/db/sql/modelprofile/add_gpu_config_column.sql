-- Add gpu_config column to model_profiles table
ALTER TABLE model_profiles 
    ADD COLUMN IF NOT EXISTS gpu_config jsonb;

-- Add index for gpu_config queries if needed
CREATE INDEX IF NOT EXISTS idx_model_profiles_gpu_config ON model_profiles USING gin(gpu_config);