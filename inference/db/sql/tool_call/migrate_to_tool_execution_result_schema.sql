-- Simple migration for tool_calls table - handles existing tables gracefully
-- Avoids DO blocks that might cause syntax issues

-- Step 1: Add missing columns (will be ignored if they already exist)
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS tool_name TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS execution_id TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS success BOOLEAN;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS args JSONB;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS result_data JSONB;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS execution_time_ms NUMERIC(10, 3);
ALTER TABLE tool_calls ADD COLUMN IF NOT EXISTS resource_usage JSONB;

-- Step 2: Update NULL values with defaults from tool_data JSONB column
UPDATE tool_calls 
SET 
    tool_name = COALESCE(tool_name, tool_data->>'tool_name', 'unknown'),
    execution_id = COALESCE(execution_id, tool_data->>'execution_id'),
    success = COALESCE(success, (tool_data->>'success')::boolean, false),
    args = COALESCE(args, tool_data->'args'),
    result_data = COALESCE(result_data, tool_data->'result_data'),
    error_message = COALESCE(error_message, tool_data->>'error_message'),
    execution_time_ms = COALESCE(execution_time_ms, 
        CASE 
            WHEN tool_data->>'execution_time_ms' IS NOT NULL 
            THEN (tool_data->>'execution_time_ms')::numeric 
            ELSE NULL 
        END),
    resource_usage = COALESCE(resource_usage, tool_data->'resource_usage')
WHERE tool_name IS NULL OR success IS NULL;