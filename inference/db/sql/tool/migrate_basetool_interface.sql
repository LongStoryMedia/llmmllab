-- Migration script to add LangChain BaseTool interface columns to existing dynamic_tools table
-- This script is idempotent and can be run multiple times safely
-- Add args_schema column for JSON schema definitions
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'args_schema') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN args_schema jsonb;
    RAISE NOTICE 'Added args_schema column to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column args_schema already exists in dynamic_tools table';
END IF;
END
$$;

-- Add return_direct column with default false
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'return_direct') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN return_direct boolean DEFAULT FALSE;
    RAISE NOTICE 'Added return_direct column to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column return_direct already exists in dynamic_tools table';
END IF;
END
$$;

-- Add tags column as text array
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'tags') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN tags text[];
    RAISE NOTICE 'Added tags column to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column tags already exists in dynamic_tools table';
END IF;
END
$$;

-- Add metadata column with default empty object
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'metadata') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN metadata jsonb DEFAULT '{}';
    RAISE NOTICE 'Added metadata column to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column metadata already exists in dynamic_tools table';
END IF;
END
$$;

-- Add handle_tool_error column for serialized error handling config
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'handle_tool_error') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN handle_tool_error text;
    RAISE NOTICE 'Added handle_tool_error column to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column handle_tool_error already exists in dynamic_tools table';
END IF;
END
$$;

-- Add handle_validation_error column for serialized validation error handling config
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'handle_validation_error') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN handle_validation_error text;
    RAISE NOTICE 'Added handle_validation_error column to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column handle_validation_error already exists in dynamic_tools table';
END IF;
END
$$;

-- Add response_format column with constraint and default
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'dynamic_tools'
            AND column_name = 'response_format') THEN
    ALTER TABLE dynamic_tools
        ADD COLUMN response_format text DEFAULT 'content';
    -- Add check constraint for valid response formats
    ALTER TABLE dynamic_tools
        ADD CONSTRAINT chk_response_format CHECK(response_format IN('content', 'content_and_artifact'));
    RAISE NOTICE 'Added response_format column with constraint to dynamic_tools table';
ELSE
    RAISE NOTICE 'Column response_format already exists in dynamic_tools table';
END IF;
END
$$;

-- Update existing rows to have default values for new columns
DO $$
BEGIN
    -- Update NULL values to defaults for boolean columns
    UPDATE
        dynamic_tools
    SET
        return_direct = FALSE
    WHERE
        return_direct IS NULL;
    -- Update NULL values to defaults for JSON columns
    UPDATE
        dynamic_tools
    SET
        metadata = '{}'::jsonb
    WHERE
        metadata IS NULL;
    -- Update NULL values to defaults for text columns
    UPDATE
        dynamic_tools
    SET
        response_format = 'content'
    WHERE
        response_format IS NULL
        OR response_format = '';
    RAISE NOTICE 'Updated existing rows with default values for new BaseTool interface columns';
END
$$;

-- Add comment to table documenting the BaseTool interface support
COMMENT ON TABLE dynamic_tools IS 'Dynamic tools table supporting LangChain BaseTool interface with embedding-based semantic search';

COMMENT ON COLUMN dynamic_tools.args_schema IS 'JSON schema for tool input validation (LangChain BaseTool interface)';

COMMENT ON COLUMN dynamic_tools.return_direct IS 'Whether tool output should be returned directly to user (LangChain BaseTool interface)';

COMMENT ON COLUMN dynamic_tools.tags IS 'Array of tags for tool categorization (LangChain BaseTool interface)';

COMMENT ON COLUMN dynamic_tools.metadata IS 'Flexible metadata object for tool information (LangChain BaseTool interface)';

COMMENT ON COLUMN dynamic_tools.handle_tool_error IS 'Serialized error handling configuration (LangChain BaseTool interface)';

COMMENT ON COLUMN dynamic_tools.handle_validation_error IS 'Serialized validation error handling configuration (LangChain BaseTool interface)';

COMMENT ON COLUMN dynamic_tools.response_format IS 'Tool response format: content or content_and_artifact (LangChain BaseTool interface)';

-- Final success message
DO $$
BEGIN
    RAISE NOTICE 'Dynamic tools table successfully migrated to support LangChain BaseTool interface';
END
$$;

