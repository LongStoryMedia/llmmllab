-- Delete message contents by message ID
DELETE FROM message_contents
WHERE message_id = $1;

