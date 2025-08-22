SELECT id, urls, topics, synthesis, created_at
FROM search_topic_syntheses
WHERE id = $1;