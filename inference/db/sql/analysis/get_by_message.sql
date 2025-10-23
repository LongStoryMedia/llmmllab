-- Get analyses by message ID
SELECT id, message_id, analysis_data, created_at
FROM analyses
WHERE message_id = $1
ORDER BY created_at ASC;