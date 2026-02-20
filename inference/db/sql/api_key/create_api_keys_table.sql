-- Create API keys table for IDE and external service integration
CREATE TABLE IF NOT EXISTS api_keys(
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id text NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  key_hash text NOT NULL UNIQUE,
  name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT NOW(),
  last_used_at timestamptz,
  expires_at timestamptz,
  is_revoked boolean NOT NULL DEFAULT FALSE,
  scopes text[] NOT NULL DEFAULT ARRAY[]::text[]
);

-- Create indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);

CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys(key_hash)
WHERE
  NOT is_revoked;

CREATE INDEX IF NOT EXISTS idx_api_keys_expires_at ON api_keys(expires_at)
WHERE
  expires_at IS NOT NULL AND NOT is_revoked;

