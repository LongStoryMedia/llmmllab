-- Add ON DELETE CASCADE functionality via triggers
CREATE OR REPLACE FUNCTION delete_related_messages_and_summaries()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM messages
  WHERE conversation_id = OLD.id;
  DELETE FROM summaries
  WHERE conversation_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS cascade_delete_trigger ON conversations;

CREATE TRIGGER cascade_delete_trigger
  BEFORE DELETE ON conversations
  FOR EACH ROW
  EXECUTE FUNCTION delete_related_messages_and_summaries();

