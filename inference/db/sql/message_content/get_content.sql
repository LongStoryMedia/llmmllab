SELECT
    message_id,
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

