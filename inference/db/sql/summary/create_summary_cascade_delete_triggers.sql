-- Create cascade delete triggers for memories when summaries are deleted
-- This ensures that when a summary is deleted, all related memories are automatically cleaned up
-- Trigger function to delete memories when a summary is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_summary_delete()
    RETURNS TRIGGER
    AS $$
BEGIN
    DELETE FROM memories
    WHERE source_id = OLD.id
        AND source = 'summary';
    RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Drop existing trigger if it exists to avoid errors on re-runs
DROP TRIGGER IF EXISTS cascade_delete_memories_on_summary ON summaries;

-- Create trigger to cascade delete related memories when summaries are deleted
CREATE TRIGGER cascade_delete_memories_on_summary
    BEFORE DELETE ON summaries
    FOR EACH ROW
    EXECUTE FUNCTION delete_memories_on_summary_delete();

