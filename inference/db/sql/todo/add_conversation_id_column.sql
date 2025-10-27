-- Migration to add conversation_id column to todos table if it doesn't exist
-- This handles the case where todos table exists but is missing the conversation_id column
-- Step 1: Check if conversation_id column exists, if not add it
DO $$
BEGIN
    IF NOT EXISTS(
        SELECT
            1
        FROM
            information_schema.columns
        WHERE
            table_name = 'todos'
            AND column_name = 'conversation_id') THEN
    -- Temporarily disable compression if it exists
    BEGIN
        PERFORM
            alter_job(job_id, scheduled => FALSE)
        FROM
            timescaledb_information.jobs
        WHERE
            proc_name = 'policy_compression'
            AND hypertable_name = 'todos';
    EXCEPTION
        WHEN OTHERS THEN
            -- Ignore errors if compression policy doesn't exist yet
            NULL;
    END;
    -- Decompress chunks if they exist
    BEGIN
        PERFORM
            decompress_chunk(c.schema_name || '.' || c.table_name)
        FROM
            timescaledb_information.chunks c
                INNER JOIN timescaledb_information.hypertables h ON h.hypertable_name = c.hypertable_name
            WHERE
                h.hypertable_name = 'todos'
                AND c.is_compressed = 't';
    EXCEPTION
        WHEN OTHERS THEN
            -- Ignore errors if no chunks are compressed or hypertable doesn't exist
            NULL;
    END;
    -- Add the missing column
    ALTER TABLE todos
        ADD COLUMN conversation_id integer;
        -- Create index for conversation-based queries
        CREATE INDEX IF NOT EXISTS idx_todos_conversation ON todos(conversation_id, created_at DESC)
        WHERE
            conversation_id IS NOT NULL;
            -- Re-enable compression if it was disabled
            BEGIN
                PERFORM
                    alter_job(job_id, scheduled => TRUE)
                FROM
                    timescaledb_information.jobs
                WHERE
                    proc_name = 'policy_compression'
                    AND hypertable_name = 'todos';
            EXCEPTION
                WHEN OTHERS THEN
                    -- Ignore errors if compression policy doesn't exist
                    NULL;
            END;
END IF;
END
$$;

