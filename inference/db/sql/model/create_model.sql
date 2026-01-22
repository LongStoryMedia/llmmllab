-- Query to create a new model
INSERT INTO models(id, name, model_name, task, modified_at, size, digest, details)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
RETURNING
    id, name, model_name, task, modified_at, size, digest, details;

