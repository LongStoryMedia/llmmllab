-- adds an image to the database
INSERT INTO images(filename, thumbnail, format, width, height, conversation_id, user_id)
    VALUES ($1, $2, $3, $4, $5, $6, $7)
RETURNING
    id;

