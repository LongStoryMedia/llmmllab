-- Search for semantically similar content
-- Parameters:
-- $1: embedding vector
-- $2: minimum similarity threshold
-- $3: limit of results
-- $4: user_id (optional, can be NULL)
-- $5: conversation_id (optional, can be NULL, has priority)
-- $6: start_date (optional, can be NULL, e.g., '2025-06-01')
-- $7: end_date (optional, can be NULL, e.g., '2025-06-05')
WITH
-- Step 1a: Find similar messages, with robust time window filter.
similar_messages_unfiltered AS (
    SELECT
        m.id AS source_id,
        m.conversation_id,
        1 -(e.embedding <=> $1) AS similarity
    FROM
        memories e
        JOIN messages m ON e.source_id = m.id
    WHERE
        e.source = 'message'
        AND 1 -(e.embedding <=> $1) > $2
        -- Filter by conversation_id if present (highest priority).
        AND ($5::bigint IS NULL
            OR m.conversation_id = $5::bigint)
            -- Add conditional time window filters with robust casting.
            AND ($6::text IS NULL
                OR m.created_at >=($6::text)::timestamptz)
            AND ($7::text IS NULL
                OR m.created_at <=($7::text)::timestamptz)
),
-- Step 1b: Find similar summaries, with the same conditional logic.
similar_summaries_unfiltered AS (
    SELECT
        s.id AS source_id,
        s.conversation_id,
        1 -(e.embedding <=> $1) AS similarity
    FROM
        memories e
        JOIN summaries s ON e.source_id = s.id
    WHERE
        e.source = 'summary'
        AND 1 -(e.embedding <=> $1) > $2
        -- Filter by conversation_id if present.
        AND ($5::bigint IS NULL
            OR s.conversation_id = $5::bigint)
            -- Add conditional time window filters with robust casting.
            AND ($6::text IS NULL
                OR s.created_at >=($6::text)::timestamptz)
            AND ($7::text IS NULL
                OR s.created_at <=($7::text)::timestamptz)
),
-- Step 1c: This CTE for user-level filtering remains the same.
filtered_convos AS (
    SELECT
        id
    FROM
        conversations
    WHERE
        user_id = $4::text
),
-- Step 1d: Apply the user filter ONLY IF conversation_id was NOT provided.
similar_messages AS (
    SELECT
        *
    FROM
        similar_messages_unfiltered
    WHERE
        -- If conversation_id is specified, this entire user check is skipped.
        $5::bigint IS NOT NULL
        OR $4::text IS NULL
        OR conversation_id IN (
            SELECT
                id
            FROM
                filtered_convos)
),
-- Step 2: Get message context for only the relevant conversations.
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
-- Step 3: Combine all results.
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
    -- And finally, any similar summaries (with the same conditional user filter)
    SELECT
        ssu.source_id,
        'summary' AS source_type,
        ssu.similarity
    FROM
        similar_summaries_unfiltered ssu
    WHERE
        -- If conversation_id is specified, this entire user check is skipped.
        $5::bigint IS NOT NULL
        OR $4::text IS NULL
        OR ssu.conversation_id IN (
            SELECT
                id
            FROM
                filtered_convos)
),
-- Step 4: Deduplicate the results.
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

