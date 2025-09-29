-- Add circuit_breaker column to model_profiles table
ALTER TABLE model_profiles
    ADD COLUMN IF NOT EXISTS circuit_breaker jsonb;

-- Add index for circuit_breaker queries if needed
CREATE INDEX IF NOT EXISTS idx_model_profiles_circuit_breaker ON model_profiles USING gin(circuit_breaker);

