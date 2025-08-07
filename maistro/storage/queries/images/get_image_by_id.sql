-- name: get_image_by_id
-- Get a specific image by ID for a user
SELECT
    id,
    filename,
    thumbnail,
    format,
    width,
    height,
    conversation_id,
    user_id,
    created_at
FROM
    images
WHERE
    id = $1
    AND user_id = $2
