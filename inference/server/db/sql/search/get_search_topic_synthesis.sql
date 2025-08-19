SELECT id, urls, topics, synthesis, created_at
FROM search_topic_synthesis
WHERE id = $1;