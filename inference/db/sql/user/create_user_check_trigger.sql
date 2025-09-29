-- Create trigger to check for valid user_id
CREATE OR REPLACE FUNCTION check_user_exists()
  RETURNS TRIGGER
  AS $$
BEGIN
  IF NOT EXISTS(
    SELECT
      1
    FROM
      users
    WHERE
      id = NEW.user_id) THEN
  RAISE EXCEPTION 'Referenced user does not exist';
END IF;
  RETURN NEW;
END;
$$
LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ensure_user_exists_trigger ON conversations;

CREATE TRIGGER ensure_user_exists_trigger
  BEFORE INSERT OR UPDATE ON conversations
  FOR EACH ROW
  EXECUTE FUNCTION check_user_exists();

