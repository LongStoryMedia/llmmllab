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
        m.role,
        m.created_at,
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
-- Step 1c: CTE for user-level filtering.
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
-- Step 2: Use LAG and LEAD to find sequential message pairs
message_context AS (
    SELECT
        sm.source_id,
        sm.conversation_id,
        sm.role,
        sm.similarity,
        sm.created_at,
        -- Get the next message ID, role, and created_at
        LEAD(sm.source_id) OVER (PARTITION BY sm.conversation_id ORDER BY sm.source_id) AS next_message_id,
        LEAD(sm.role) OVER (PARTITION BY sm.conversation_id ORDER BY sm.source_id) AS next_message_role,
    LEAD(sm.created_at) OVER (PARTITION BY sm.conversation_id ORDER BY sm.source_id) AS next_message_created_at,
    -- Get the previous message ID, role, and created_at
    LAG(sm.source_id) OVER (PARTITION BY sm.conversation_id ORDER BY sm.source_id) AS prev_message_id,
    LAG(sm.role) OVER (PARTITION BY sm.conversation_id ORDER BY sm.source_id) AS prev_message_role,
    LAG(sm.created_at) OVER (PARTITION BY sm.conversation_id ORDER BY sm.source_id) AS prev_message_created_at
FROM
    similar_messages sm
ORDER BY
    sm.conversation_id,
    sm.source_id
),
message_pairs AS (
    -- Find user messages with their next assistant response
    SELECT
        mc.source_id AS first_message_id,
        'user' AS first_message_role,
        mc.created_at AS first_message_created_at,
        mc.next_message_id AS second_message_id,
        'assistant' AS second_message_role,
        mc.next_message_created_at AS second_message_created_at,
        mc.conversation_id,
        mc.similarity,
        'user_first' AS pair_type -- Mark that user message comes first
    FROM
        message_context mc
    WHERE
        mc.role = 'user'
        AND mc.next_message_role = 'assistant'
        AND mc.next_message_id IS NOT NULL
    UNION ALL
    -- Find assistant messages with their previous user query
    SELECT
        mc.prev_message_id AS first_message_id,
        'user' AS first_message_role,
        mc.prev_message_created_at AS first_message_created_at,
        mc.source_id AS second_message_id,
        'assistant' AS second_message_role,
        mc.created_at AS second_message_created_at,
        mc.conversation_id,
        mc.similarity,
        'assistant_match' AS pair_type -- Mark that assistant message was the match
    FROM
        message_context mc
    WHERE
        mc.role = 'assistant'
        AND mc.prev_message_role = 'user'
        AND mc.prev_message_id IS NOT NULL
),
-- Step 3: Deduplicate message pairs by prioritizing pairs with higher similarity
-- and ensuring we don't include the same message in multiple pairs
deduplicated_message_pairs AS (
    SELECT
        first_message_id,
        first_message_role,
        first_message_created_at,
        second_message_id,
        second_message_role,
        second_message_created_at,
        conversation_id,
        similarity,
        pair_type,
        -- For each pair of messages, keep the one with the highest similarity
        ROW_NUMBER() OVER (PARTITION BY first_message_id,
            second_message_id ORDER BY similarity DESC) AS exact_pair_rank,
        -- For each message, keep only one pair it belongs to (the highest similarity)
        ROW_NUMBER() OVER (PARTITION BY first_message_id ORDER BY similarity DESC) AS first_message_rank,
        ROW_NUMBER() OVER (PARTITION BY second_message_id ORDER BY similarity DESC) AS second_message_rank
    FROM
        message_pairs
),
-- Step 4: Prepare message pairs to fetch with their similarity scores
-- Always putting user messages first, assistant messages second
message_results_to_fetch AS (
    -- Add first message (user)
    SELECT
        first_message_id AS source_id,
        'message' AS source_type,
        similarity,
        conversation_id,
        1 AS pair_order, -- User message first
        similarity AS original_similarity,
        CONCAT(first_message_id, '-', second_message_id) AS pair_key -- Create unique pair key
    FROM
        deduplicated_message_pairs
    WHERE
        exact_pair_rank = 1 -- Only the highest similarity for this exact pair
        AND first_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND second_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND first_message_role = 'user' -- Verify it's a user message
        AND second_message_role = 'assistant' -- Verify it's paired with an assistant message
    UNION ALL
    -- Add second message (assistant)
    SELECT
        second_message_id AS source_id,
        'message' AS source_type,
        similarity, -- Use same similarity as original
        conversation_id,
        2 AS pair_order, -- Assistant message second
        similarity AS original_similarity,
        CONCAT(first_message_id, '-', second_message_id) AS pair_key -- Same pair key to match
    FROM
        deduplicated_message_pairs
    WHERE
        exact_pair_rank = 1 -- Only the highest similarity for this exact pair
        AND first_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND second_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND first_message_role = 'user' -- Verify it's a user message
        AND second_message_role = 'assistant' -- Verify it's paired with an assistant message
),
-- Step 5: Include the summaries
summary_results_to_fetch AS (
    SELECT
        ssu.source_id,
        'summary' AS source_type,
        ssu.similarity,
        ssu.conversation_id,
        0 AS pair_order, -- Summaries are standalone
        ssu.similarity AS original_similarity,
        CONCAT('summary-', ssu.source_id) AS pair_key -- Each summary is its own group
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
-- Step 6: Combine message pairs and summaries
all_results_to_fetch AS (
    SELECT
        *
    FROM
        message_results_to_fetch
    UNION ALL
    SELECT
        *
    FROM
        summary_results_to_fetch
),
-- Step 7: Prepare final results
-- Keep the original ordering and uniqueness ensured from previous steps
unique_results AS (
    SELECT
        source_id,
        source_type,
        similarity,
        conversation_id,
        pair_order,
        pair_key,
        original_similarity,
        -- Generate a rank for the complete dataset to aid in LIMIT application
        ROW_NUMBER() OVER (ORDER BY original_similarity DESC,
            pair_key,
            pair_order) AS global_rank
FROM
    all_results_to_fetch
),
-- Step 8: Fetch the final content with proper ordering
-- First apply LIMIT to pairs (by their highest similarity), then ensure both messages in each pair are included
limited_pairs AS (
    -- Use a simple approach to limit the number of pairs
    -- Get the top N pairs based on similarity
    SELECT
        pair_key
    FROM
        unique_results
    GROUP BY
        pair_key
    ORDER BY
        MAX(similarity) DESC
    LIMIT $3
)
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
    u.pair_key IN (
        SELECT
            pair_key
        FROM
            limited_pairs)
ORDER BY
    u.similarity DESC, -- Sort by highest similarity first
    COALESCE(m.conversation_id, s.conversation_id), -- Keep conversation pairs together
    COALESCE(m.created_at, s.created_at) -- Maintain chronological order within conversations
