-- Add draft_model column to model_profiles table
ALTER TABLE model_profiles
    ADD COLUMN IF NOT EXISTS draft_model text;

-- Add comment for documentation
COMMENT ON COLUMN model_profiles.draft_model IS 'Optional draft model to use for faster initial responses';

