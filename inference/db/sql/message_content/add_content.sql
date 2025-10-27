-- Add a new message content (idempotent - safe to run multiple times)
INSERT INTO message_contents (
    message_id,
    type,
    text_content,
    url,
    created_at
) VALUES (
    $1, $2, $3, $4, $5
)
ON CONFLICT (message_id, type, COALESCE(text_content, ''), COALESCE(url, ''))
DO UPDATE SET 
    created_at = COALESCE(message_contents.created_at, EXCLUDED.created_at)
RETURNING id;
