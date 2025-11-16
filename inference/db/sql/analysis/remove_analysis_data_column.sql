-- Remove the analysis_data column from analyses table
-- This column was used for legacy JSONB storage but is no longer needed
-- with the new schema-first approach

-- Step 1: Drop any indexes on the analysis_data column
DROP INDEX IF EXISTS idx_analyses_data;

-- Step 2: Drop the analysis_data column
ALTER TABLE analyses DROP COLUMN IF EXISTS analysis_data;

-- Verification: The table should now only contain the structured columns
-- without the legacy analysis_data JSONB column