-- Delete analyses by message ID
DELETE FROM analyses
WHERE message_id = $1;

