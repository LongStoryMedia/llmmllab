-- Search for semantically similar content within a specific conversation
-- Parameters:
-- $1: conversation_id
-- $2: embedding vector
-- $3: limit of results
WITH similar_messages AS (
  -- Find similar messages
  SELECT
    m.id AS source_id,
    m.role,
    m.content,
    m.conversation_id,
    1 -(e.embedding <=> $2) AS similarity,
    'message' AS source_type,
    e.created_at AS created_at
  FROM
    memories e
    JOIN messages m ON e.source_id = m.id
  WHERE
    m.conversation_id = $1
    AND 1 -(e.embedding <=> $2) > $4
    AND e.source = 'message'
),
paired_messages AS (
  -- Find message pairs (user -> assistant, assistant -> user)
  SELECT
    sm.source_id,
    sm.role,
    sm.content,
    sm.conversation_id,
    sm.similarity,
    sm.source_type,
    sm.created_at,
    row_number() OVER (PARTITION BY sm.source_id ORDER BY sm.similarity DESC) AS rn
  FROM
    similar_messages sm
  UNION ALL
  -- Add next messages for user messages (to get the assistant response)
  SELECT
    m.id AS source_id,
    m.role,
    m.content,
    m.conversation_id,
    sm.similarity, -- Keep original similarity score
    'message' AS source_type,
    m.created_at,
    2 AS rn -- Mark as paired message
  FROM
    similar_messages sm
    JOIN messages m ON sm.conversation_id = m.conversation_id
  WHERE
    sm.role = 'user'
    AND m.role = 'assistant'
    AND m.created_at > sm.created_at -- Next message in conversation
    AND m.conversation_id = sm.conversation_id
    AND NOT EXISTS (
      SELECT
        1
      FROM
        messages m2
      WHERE
        m2.conversation_id = sm.conversation_id
        AND m2.created_at > sm.created_at
        AND m2.created_at < m.created_at) -- Ensure it's the immediately following message
    UNION ALL
    -- Add previous messages for assistant messages (to get the user query)
    SELECT
      m.id AS source_id,
      m.role,
      m.content,
      m.conversation_id,
      sm.similarity, -- Keep original similarity score
      'message' AS source_type,
      m.created_at,
      2 AS rn -- Mark as paired message
    FROM
      similar_messages sm
      JOIN messages m ON sm.conversation_id = m.conversation_id
    WHERE
      sm.role = 'assistant'
      AND m.role = 'user'
      AND m.created_at < sm.created_at -- Previous message in conversation
      AND m.conversation_id = sm.conversation_id
      AND NOT EXISTS (
        SELECT
          1
        FROM
          messages m2
        WHERE
          m2.conversation_id = sm.conversation_id
          AND m2.created_at < sm.created_at
          AND m2.created_at > m.created_at) -- Ensure it's the immediately preceding message
),
unified_results AS (
  -- Include paired messages first
  SELECT
    source_id,
    ROLE,
    content,
    conversation_id,
    similarity,
    source_type,
    created_at
  FROM
    paired_messages
  UNION ALL
  -- Find similar summaries
  SELECT
    s.id AS source_id,
    'system' AS role,
    s.content,
    s.conversation_id,
    1 -(e.embedding <=> $2) AS similarity,
  'summary' AS source_type,
  e.created_at AS created_at
FROM
  memories e
  JOIN summaries s ON e.source_id = s.id
  WHERE
    s.conversation_id = $1
    AND 1 -(e.embedding <=> $2) > $4
    AND e.source = 'summary'
    -- UNION ALL
    -- -- Find similar research results
    -- SELECT
    --   r.id,
    --   'assistant' AS role,
    --   CAST(jsonb_array_elements(r.results) ->> 'content' AS text) AS content,
    --   r.conversation_id,
    --   1 -(e.embedding <=> $2) AS similarity,
    --   'research' AS source_type
    -- FROM
    --   memories e
    --   JOIN research_tasks r ON e.source_id = r.id
    -- WHERE
    --   r.conversation_id = $1
    --   AND e.source = 'research'
    --   AND r.results IS NOT NULL
)
SELECT
  role,
  source_id,
  content,
  source_type,
  similarity,
  conversation_id,
  created_at
FROM
  unified_results
ORDER BY
  similarity,
  created_at DESC
LIMIT $3;

