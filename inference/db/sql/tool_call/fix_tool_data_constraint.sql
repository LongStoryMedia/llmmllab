-- Migration: Remove NOT NULL constraint from tool_calls.tool_data column
-- This fixes TimescaleDB hypertable constraint violations that cause transaction rollbacks

-- Remove NOT NULL constraint on legacy tool_data column
ALTER TABLE tool_calls ALTER COLUMN tool_data DROP NOT NULL;

-- Add comment to document this is a legacy column
COMMENT ON COLUMN tool_calls.tool_data IS 'Legacy column - NOT NULL constraint removed to prevent hypertable violations';