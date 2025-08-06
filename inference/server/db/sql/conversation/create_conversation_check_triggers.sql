-- Add triggers to maintain referential integrity between conversations and messages
CREATE OR REPLACE FUNCTION check_conversation_exists()
  RETURNS TRIGGER
  AS $$
BEGIN
  IF NOT EXISTS(
    SELECT
      1
    FROM
      conversations
    WHERE
      id = NEW.conversation_id) THEN
  RAISE EXCEPTION 'Referenced conversation does not exist';
END IF;
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ensure_conversation_exists_messages_trigger ON messages;

CREATE TRIGGER ensure_conversation_exists_messages_trigger
  BEFORE INSERT OR UPDATE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION check_conversation_exists();

DROP TRIGGER IF EXISTS ensure_conversation_exists_summaries_trigger ON summaries;

CREATE TRIGGER ensure_conversation_exists_summaries_trigger
  BEFORE INSERT OR UPDATE ON summaries
  FOR EACH ROW
  EXECUTE FUNCTION check_conversation_exists();

