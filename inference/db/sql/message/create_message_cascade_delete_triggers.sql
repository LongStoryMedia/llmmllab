-- Create cascade delete triggers for message-related data
-- This ensures that when a message is deleted, all related data is automatically cleaned up
-- Trigger function to delete message_contents when a message is deleted
CREATE OR REPLACE FUNCTION delete_message_contents_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM message_contents
  WHERE message_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Trigger function to delete thoughts when a message is deleted
CREATE OR REPLACE FUNCTION delete_thoughts_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM thoughts
  WHERE message_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Trigger function to delete tool_calls when a message is deleted
CREATE OR REPLACE FUNCTION delete_tool_calls_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM tool_calls
  WHERE message_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Trigger function to delete analyses when a message is deleted
CREATE OR REPLACE FUNCTION delete_analyses_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM analyses
  WHERE message_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Trigger function to delete documents when a message is deleted
CREATE OR REPLACE FUNCTION delete_documents_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM documents
  WHERE message_id = OLD.id;
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Trigger function to delete memories when a message is deleted
CREATE OR REPLACE FUNCTION delete_memories_on_message_delete()
  RETURNS TRIGGER
  AS $$
BEGIN
  DELETE FROM memories
  WHERE source_id = OLD.id
    AND source = 'message';
  RETURN OLD;
END;
$$
LANGUAGE plpgsql;

-- Drop existing triggers if they exist to avoid errors on re-runs
DROP TRIGGER IF EXISTS cascade_delete_message_contents_on_message ON messages;

DROP TRIGGER IF EXISTS cascade_delete_thoughts_on_message ON messages;

DROP TRIGGER IF EXISTS cascade_delete_tool_calls_on_message ON messages;

DROP TRIGGER IF EXISTS cascade_delete_analyses_on_message ON messages;

DROP TRIGGER IF EXISTS cascade_delete_documents_on_message ON messages;

DROP TRIGGER IF EXISTS cascade_delete_memories_on_message ON messages;

-- Create triggers to cascade delete related data when messages are deleted
CREATE TRIGGER cascade_delete_message_contents_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_message_contents_on_message_delete();

CREATE TRIGGER cascade_delete_thoughts_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_thoughts_on_message_delete();

CREATE TRIGGER cascade_delete_tool_calls_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_tool_calls_on_message_delete();

CREATE TRIGGER cascade_delete_analyses_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_analyses_on_message_delete();

CREATE TRIGGER cascade_delete_documents_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_documents_on_message_delete();

CREATE TRIGGER cascade_delete_memories_on_message
  BEFORE DELETE ON messages
  FOR EACH ROW
  EXECUTE FUNCTION delete_memories_on_message_delete();

