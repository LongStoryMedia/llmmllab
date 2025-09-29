-- Delete a model profile by ID and user ID
DELETE FROM model_profiles
WHERE id = $1 AND user_id = $2
