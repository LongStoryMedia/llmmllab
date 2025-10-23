-- Add analysis to database
INSERT INTO analyses (
    message_id, 
    analysis_data, 
    created_at
)
VALUES ($1, $2, COALESCE($3, NOW()))
RETURNING id;