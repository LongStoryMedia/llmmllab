-- Update conversation updated_at when a message is added
CREATE OR REPLACE FUNCTION update_conversation_updated_at()
  RETURNS TRIGGER
  AS $$
BEGIN
  UPDATE
    conversations
  SET
    updated_at = NOW()
  WHERE
    id = NEW.conversation_id;
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_conversation_updated_at_trigger ON messages;

CREATE TRIGGER update_conversation_updated_at_trigger
  AFTER INSERT ON messages
  FOR EACH ROW
  EXECUTE FUNCTION update_conversation_updated_at();

