-- Delete thoughts by message ID
DELETE FROM thoughts
WHERE message_id = $1;

