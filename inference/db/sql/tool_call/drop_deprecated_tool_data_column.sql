-- Remove deprecated tool_data column after migration to structured schema
-- This should be run after migrate_to_tool_execution_result_schema.sql
-- and after verifying all data has been properly migrated

-- Drop the deprecated tool_data column
ALTER TABLE tool_calls DROP COLUMN IF EXISTS tool_data;

-- Verify the table structure is clean
SELECT 
    column_name, 
    data_type, 
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_name = 'tool_calls' 
    AND table_schema = current_schema()
ORDER BY ordinal_position;