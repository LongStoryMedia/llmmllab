-- Migration to update tool_calls table schema to match ToolExecutionResult model
-- This migration extracts data from the generic tool_data JSONB column into specific columns

-- Step 1: Add new columns to match ToolExecutionResult fields
ALTER TABLE tool_calls 
ADD COLUMN IF NOT EXISTS tool_name TEXT,
ADD COLUMN IF NOT EXISTS execution_id TEXT,
ADD COLUMN IF NOT EXISTS success BOOLEAN,
ADD COLUMN IF NOT EXISTS args JSONB,
ADD COLUMN IF NOT EXISTS result_data JSONB,
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS execution_time_ms NUMERIC(10, 3) CHECK (execution_time_ms >= 0),
ADD COLUMN IF NOT EXISTS resource_usage JSONB;

-- Step 2: Migrate existing data from tool_data JSONB to new columns
-- Only run this if there are existing rows and the new columns are empty
UPDATE tool_calls 
SET 
    tool_name = COALESCE(tool_data->>'tool_name', 'unknown'),
    execution_id = tool_data->>'execution_id',
    success = COALESCE((tool_data->>'success')::boolean, false),
    args = tool_data->'args',
    result_data = tool_data->'result_data',
    error_message = tool_data->>'error_message',
    execution_time_ms = CASE 
        WHEN tool_data->>'execution_time_ms' IS NOT NULL 
        THEN (tool_data->>'execution_time_ms')::numeric 
        ELSE NULL 
    END,
    resource_usage = tool_data->'resource_usage'
WHERE tool_name IS NULL;

-- Step 3: Make required columns NOT NULL after data migration
ALTER TABLE tool_calls 
ALTER COLUMN tool_name SET NOT NULL,
ALTER COLUMN success SET NOT NULL;

-- Step 4: Create new indexes for the structured columns
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_success ON tool_calls(success);
CREATE INDEX IF NOT EXISTS idx_tool_calls_execution_id ON tool_calls(execution_id) WHERE execution_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tool_calls_args ON tool_calls USING GIN (args) WHERE args IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tool_calls_result_data ON tool_calls USING GIN (result_data) WHERE result_data IS NOT NULL;

-- Step 5: Drop the old generic tool_data column (optional - keep for backwards compatibility)
-- ALTER TABLE tool_calls DROP COLUMN IF EXISTS tool_data;

-- Note: We keep tool_data column for now to maintain backwards compatibility
-- It can be dropped in a future migration once all code is updated