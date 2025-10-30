-- Simple migration for tool_calls table - handles existing tables gracefully
-- Avoids DO blocks that might cause syntax issues
-- Step 1: Add missing columns (will be ignored if they already exist)
ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS tool_name text;

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS execution_id text;

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS success boolean;

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS args jsonb;

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS result_data jsonb;

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS error_message text;

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS execution_time_ms numeric(10, 3);

ALTER TABLE tool_calls
    ADD COLUMN IF NOT EXISTS resource_usage jsonb;

-- Step 2: Update NULL values with defaults from tool_data JSONB column (if it exists)
-- Check if tool_data column exists before trying to migrate from it
DO $$
BEGIN
    -- Only attempt migration if tool_data column exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'tool_calls' 
        AND column_name = 'tool_data'
        AND table_schema = current_schema()
    ) THEN
        -- Migrate data from tool_data column
        UPDATE tool_calls
        SET
            tool_name = COALESCE(tool_name, tool_data ->> 'tool_name', 'unknown'),
            execution_id = COALESCE(execution_id, tool_data ->> 'execution_id'),
            success = COALESCE(success, (tool_data ->> 'success')::boolean, FALSE),
            args = COALESCE(args, tool_data -> 'args'),
            result_data = COALESCE(result_data, tool_data -> 'result_data'),
            error_message = COALESCE(error_message, tool_data ->> 'error_message'),
            execution_time_ms = COALESCE(execution_time_ms, 
                CASE WHEN tool_data ->> 'execution_time_ms' IS NOT NULL THEN
                    (tool_data ->> 'execution_time_ms')::numeric
                ELSE
                    NULL
                END),
            resource_usage = COALESCE(resource_usage, tool_data -> 'resource_usage')
        WHERE
            tool_name IS NULL
            OR success IS NULL;
        
        RAISE NOTICE 'Migrated data from tool_data column to new schema';
    ELSE
        -- Set default values for any NULL fields if no migration is needed
        UPDATE tool_calls
        SET
            tool_name = COALESCE(tool_name, 'unknown'),
            success = COALESCE(success, FALSE)
        WHERE
            tool_name IS NULL
            OR success IS NULL;
        
        RAISE NOTICE 'No tool_data column found - applied default values only';
    END IF;
END $$;

