UPDATE
  conversations
SET
  updated_at = NOW()
WHERE
  id = $1
