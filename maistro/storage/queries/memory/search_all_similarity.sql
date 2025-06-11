-- Search for semantically similar content across all conversations
-- Parameters:
-- $1: embedding vector
-- $2: minimum similarity threshold
-- $3: limit of results
-- $4: user_id (optional, can be NULL)
WITH
-- Step 1: Find the initial set of similar messages. This is the main driver.
-- We get the ID, similarity, and crucially, the conversation_id to scope the next step.
similar_messages AS (
  SELECT
    m.id AS source_id,
    m.conversation_id,
    1 -(e.embedding <=> $1) AS similarity
  FROM
    memories e
    JOIN messages m ON e.source_id = m.id
    JOIN conversations c ON m.conversation_id = c.id
  WHERE
    e.source = 'message'
    AND 1 -(e.embedding <=> $1) > $2
    AND ($4::text IS NULL
      OR c.user_id = $4::text)
),
-- Step 2: For ONLY the conversations found above, get a full message list
-- with the immediate next/previous message IDs pre-calculated.
-- This is far more efficient than the original's multiple NOT EXISTS subqueries.
message_context AS (
  SELECT
    m.id,
    m.role,
    LEAD(m.id) OVER (PARTITION BY m.conversation_id ORDER BY m.created_at) AS next_message_id,
    LAG(m.id) OVER (PARTITION BY m.conversation_id ORDER BY m.created_at) AS prev_message_id
  FROM
    messages m
  WHERE
    m.conversation_id IN (
      SELECT
        conversation_id
      FROM
        similar_messages)
),
-- Step 3: Combine all the message IDs we want to retrieve:
-- the original, its pair, and also add in any similar summaries.
results_to_fetch AS (
  -- The original similar messages
  SELECT
    sm.source_id,
    'message' AS source_type,
    sm.similarity
  FROM
    similar_messages sm
  UNION ALL
  -- The user's query that led to a similar assistant response
  SELECT
    mc.prev_message_id,
    'message' AS source_type,
    sm.similarity
  FROM
    similar_messages sm
    JOIN message_context mc ON sm.source_id = mc.id
  WHERE
    mc.role = 'assistant'
    AND mc.prev_message_id IS NOT NULL
  UNION ALL
  -- The assistant's response to a similar user query
  SELECT
    mc.next_message_id,
    'message' AS source_type,
    sm.similarity
  FROM
    similar_messages sm
    JOIN message_context mc ON sm.source_id = mc.id
  WHERE
    mc.role = 'user'
    AND mc.next_message_id IS NOT NULL
  UNION ALL
  -- And finally, any similar summaries
  SELECT
    s.id AS source_id,
    'summary' AS source_type,
    1 -(e.embedding <=> $1) AS similarity
  FROM
    memories e
    JOIN summaries s ON e.source_id = s.id
    JOIN conversations c ON s.conversation_id = c.id
  WHERE
    e.source = 'summary'
    AND 1 -(e.embedding <=> $1) > $2
    AND ($4::text IS NULL
      OR c.user_id = $4::text)
),
-- Step 4: Deduplicate the results. A message could be found directly
-- and also be a "pair" to another message. We prioritize the one with higher similarity.
unique_results AS (
  SELECT
    source_id,
    source_type,
    similarity,
    ROW_NUMBER() OVER (PARTITION BY source_id,
      source_type ORDER BY similarity DESC) AS row_num
  FROM
    results_to_fetch)
  -- Step 5: Fetch the final content, order, and limit the results.
  -- Joining the content tables at the end reduces the amount of data being passed around.
  SELECT
    COALESCE(m.role, 'system') AS role,
  u.source_id,
  COALESCE(m.content, s.content) AS content,
  u.source_type,
  u.similarity,
  COALESCE(m.conversation_id, s.conversation_id) AS conversation_id,
  COALESCE(m.created_at, s.created_at) AS created_at
FROM
  unique_results u
  LEFT JOIN messages m ON u.source_id = m.id
    AND u.source_type = 'message'
  LEFT JOIN summaries s ON u.source_id = s.id
    AND u.source_type = 'summary'
WHERE
  u.row_num = 1
ORDER BY
  u.similarity DESC,
  created_at DESC
LIMIT $3;

