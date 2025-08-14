SELECT
    message_id,
    message_created_at,
    created_at,
    type,
    text_content,
    url
FROM
    message_contents
WHERE
    message_id = $1
ORDER BY
    id ASC;

