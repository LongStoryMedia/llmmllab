-- Create cascade delete triggers for memories when search topic syntheses are deleted
-- This ensures that when a search topic synthesis is deleted, all related memories are automatically cleaned up
-- Trigger function to delete memories when a search topic synthesis is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_search_delete()
    RETURNS TRIGGER
    AS $$
BEGIN
    DELETE FROM memories
    WHERE source_id = OLD.id
        AND source = 'search';
    RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Drop existing trigger if it exists to avoid errors on re-runs
DROP TRIGGER IF EXISTS cascade_delete_memories_on_search ON search_topic_syntheses;

-- Create trigger to cascade delete related memories when search topic syntheses are deleted
CREATE TRIGGER cascade_delete_memories_on_search
    BEFORE DELETE ON search_topic_syntheses
    FOR EACH ROW
    EXECUTE FUNCTION delete_memories_on_search_delete();

