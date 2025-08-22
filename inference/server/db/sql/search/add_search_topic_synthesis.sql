INSERT INTO search_topic_syntheses(urls, topics, synthesis, conversation_id, created_at)
    VALUES ($1, $2, $3, $4, NOW())
RETURNING
    id;

