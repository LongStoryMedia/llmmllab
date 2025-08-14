-- Trigger function to delete memories when a message is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM memories
  WHERE source = 'message'
    AND source_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cascade_delete_memories_on_message ON messages;

CREATE TRIGGER cascade_delete_memories_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_memories_on_message_delete();

-- Trigger function to delete memories when a summary is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_summary_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM memories
  WHERE source = 'summary'
    AND source_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cascade_delete_memories_on_summary ON summaries;

CREATE TRIGGER cascade_delete_memories_on_summary
  BEFORE DELETE ON summaries
  FOR EACH ROW
  EXECUTE FUNCTION delete_memories_on_summary_delete();

