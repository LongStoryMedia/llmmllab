-- Create model_profiles table for storing model profiles
CREATE TABLE IF NOT EXISTS model_profiles(
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL,
  name text NOT NULL,
  description text,
  model_name text,
  parameters jsonb,
  system_prompt text,
  model_version text,
  type INTEGER,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  updated_at timestamptz NOT NULL DEFAULT NOW()
);

