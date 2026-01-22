-- Drop the verbose column from dynamic_tools table if it exists
-- This script is idempotent and can be run multiple times safely
DO $$
BEGIN
    IF EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'verbose') THEN
    ALTER TABLE dynamic_tools
        DROP COLUMN VERBOSE;
    RAISE NOTICE 'Dropped verbose column from dynamic_tools table';
ELSE
    RAISE NOTICE 'Column verbose does not exist in dynamic_tools table';
END IF;
END
$$;

