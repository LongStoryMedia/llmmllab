-- Delete message contents first (child table)
DELETE FROM message_contents
WHERE message_id = $1;

-- Then delete the message (parent table)
DELETE FROM messages
WHERE id = $1;