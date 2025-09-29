-- Query to get a specific model by ID
SELECT
    id,
    name,
    model_name,
    task,
    modified_at,
    size,
    digest,
    details
FROM
    models
WHERE
    id = $1;

