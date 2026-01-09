-- Search for semantically similar content
-- Parameters:
-- $1: embedding vector
-- $2: minimum similarity threshold
-- $3: limit of results
-- $4: user_id (optional, can be NULL)
-- $5: conversation_id (optional, can be NULL)
-- $6: start_date (optional, can be NULL, e.g., '2025-06-01')
-- $7: end_date (optional, can be NULL, e.g., '2025-06-05')
WITH
-- Step 1a: Find similar messages.
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
        -- Filter by user_id directly on memories table
        AND ($4::text IS NULL
            OR e.user_id = $4::text)
            -- Add conditional time window filters.
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
        -- Filter by user_id directly on memories table
        AND ($4::text IS NULL
            OR e.user_id = $4::text)
            -- Add conditional time window filters.
            AND ($6::text IS NULL
                OR s.created_at >=($6::text)::timestamptz)
            AND ($7::text IS NULL
                OR s.created_at <=($7::text)::timestamptz)
),
-- Step 1c: Find similar search_topic_syntheses
similar_search_topics_unfiltered AS (
    SELECT
        st.id AS source_id,
        st.conversation_id,
        1 -(e.embedding <=> $1) AS similarity
    FROM
        memories e
        JOIN search_topic_syntheses st ON e.source_id = st.id
    WHERE
        e.source = 'search'
        AND 1 -(e.embedding <=> $1) > $2
        -- Filter by user_id directly on memories table
        AND ($4::text IS NULL
            OR e.user_id = $4::text)
            -- Add conditional time window filters.
            AND ($6::text IS NULL
                OR st.created_at >=($6::text)::timestamptz)
            AND ($7::text IS NULL
                OR st.created_at <=($7::text)::timestamptz)
),
-- Step 1d: Find similar file documents
similar_documents_unfiltered AS (
    SELECT
        d.id AS source_id,
        m.conversation_id,
        1 -(e.embedding <=> $1) AS similarity
    FROM
        memories e
        JOIN documents d ON e.source_id = d.id
        JOIN messages m ON d.message_id = m.id
    WHERE
        e.source = 'document'
        AND 1 -(e.embedding <=> $1) > $2
        -- Filter by user_id directly on memories table
        AND ($4::text IS NULL
            OR e.user_id = $4::text)
            -- Add conditional time window filters.
            AND ($6::text IS NULL
                OR d.created_at >=($6::text)::timestamptz)
            AND ($7::text IS NULL
                OR d.created_at <=($7::text)::timestamptz)
),
-- Step 1e: Apply conversation filtering only (user filtering already applied above)
similar_messages AS (
    SELECT
        *
    FROM
        similar_messages_unfiltered
    WHERE
        -- If no conversation_id provided, show all conversations
($5::integer IS NULL)
        OR
        -- If conversation_id provided, filter by specific conversation
(conversation_id = $5::integer)
),
-- Step 2: Find user+assistant message pairs
message_pairs AS (
    -- For user messages: find the next assistant message in same conversation
    SELECT
        user_msg.source_id AS first_message_id,
        'user' AS first_message_role,
        user_msg.created_at AS first_message_created_at,
(
            SELECT
                m.id
            FROM
                messages m
            WHERE
                m.conversation_id = user_msg.conversation_id
                AND m.role = 'assistant'
                AND m.created_at > user_msg.created_at
            ORDER BY
                m.created_at ASC
            LIMIT 1) AS second_message_id,
        'assistant' AS second_message_role,
(
            SELECT
                m.created_at
            FROM
                messages m
            WHERE
                m.conversation_id = user_msg.conversation_id
                AND m.role = 'assistant'
                AND m.created_at > user_msg.created_at
            ORDER BY
                m.created_at ASC
            LIMIT 1) AS second_message_created_at,
        user_msg.conversation_id,
        user_msg.similarity,
        'user_matched' AS pair_type
    FROM
        similar_messages user_msg
    WHERE
        user_msg.role = 'user'
    UNION ALL
    -- For assistant messages: find the previous user message in same conversation
    SELECT
        (
            SELECT
                m.id
            FROM
                messages m
            WHERE
                m.conversation_id = assistant_msg.conversation_id
                AND m.role = 'user'
                AND m.created_at < assistant_msg.created_at
            ORDER BY
                m.created_at DESC
            LIMIT 1) AS first_message_id,
        'user' AS first_message_role,
(
            SELECT
                m.created_at
            FROM
                messages m
            WHERE
                m.conversation_id = assistant_msg.conversation_id
                AND m.role = 'user'
                AND m.created_at < assistant_msg.created_at
            ORDER BY
                m.created_at DESC
            LIMIT 1) AS first_message_created_at,
        assistant_msg.source_id AS second_message_id,
        'assistant' AS second_message_role,
        assistant_msg.created_at AS second_message_created_at,
        assistant_msg.conversation_id,
        assistant_msg.similarity,
        'assistant_matched' AS pair_type
    FROM
        similar_messages assistant_msg
    WHERE
        assistant_msg.role = 'assistant'
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
    -- Add first message (user) from pairs
    SELECT
        first_message_id AS source_id,
        'message' AS source_type,
        similarity,
        conversation_id,
        1 AS pair_order, -- User message first
        similarity AS original_similarity,
        CONCAT(COALESCE(first_message_id::text, 'null'), '-', COALESCE(second_message_id::text, 'null')) AS pair_key -- Create unique pair key
    FROM
        deduplicated_message_pairs
    WHERE
        exact_pair_rank = 1 -- Only the highest similarity for this exact pair
        AND first_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND first_message_id IS NOT NULL -- Must have a first message
        AND (
            -- Only complete pairs: user + assistant
            second_message_id IS NOT NULL
            AND second_message_rank = 1
            AND first_message_role = 'user'
            AND second_message_role = 'assistant')
    UNION ALL
    -- Add second message (assistant) from pairs (only for actual pairs, not fallback singles)
    SELECT
        second_message_id AS source_id,
        'message' AS source_type,
        similarity, -- Use same similarity as original
        conversation_id,
        2 AS pair_order, -- Assistant message second
        similarity AS original_similarity,
        CONCAT(COALESCE(first_message_id::text, 'null'), '-', COALESCE(second_message_id::text, 'null')) AS pair_key -- Same pair key to match
    FROM
        deduplicated_message_pairs
    WHERE
        exact_pair_rank = 1 -- Only the highest similarity for this exact pair
        AND first_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND second_message_rank = 1 -- Only include this message in one pair (highest similarity)
        AND first_message_id IS NOT NULL -- Must have a first message
        AND second_message_id IS NOT NULL -- Must have a second message for pairs
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
    -- Apply conversation filtering for summaries (user filtering already applied)
($5::integer IS NULL)
    OR (ssu.conversation_id = $5::integer)
),
-- step 6, Include the search topic syntheses
search_results_to_fetch AS (
    SELECT
        ss.source_id,
        'search' AS source_type,
        ss.similarity,
        ss.conversation_id,
        0 AS pair_order, -- Search results are standalone
        ss.similarity AS original_similarity,
        CONCAT('search-', ss.source_id) AS pair_key -- Each search result is its own group
    FROM
        similar_search_topics_unfiltered ss
    WHERE
    -- Apply conversation filtering for search topics (user filtering already applied)
($5::integer IS NULL)
    OR (ss.conversation_id = $5::integer)
),
-- step 7, Include the file documents
document_results_to_fetch AS (
    SELECT
        sa.source_id,
        'document' AS source_type,
        sa.similarity,
        sa.conversation_id,
        0 AS pair_order, -- documents are standalone
        sa.similarity AS original_similarity,
        CONCAT('document-', sa.source_id) AS pair_key -- Each document is its own group
    FROM
        similar_documents_unfiltered sa
    WHERE
    -- Apply conversation filtering for documents (user filtering already applied)
($5::integer IS NULL)
    OR (sa.conversation_id = $5::integer)
),
-- Step 8: Combine message pairs, summaries, search results, and documents
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
    UNION ALL
    SELECT
        *
    FROM
        search_results_to_fetch
    UNION ALL
    SELECT
        *
    FROM
        document_results_to_fetch
),
-- Step 9: Prepare final results
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
-- Step 9: Fetch the final content with proper ordering
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
    COALESCE(mc.text_content, s.content, ss.synthesis, d.text_content, d.filename) AS content,
    u.source_type,
    u.similarity,
    COALESCE(m.conversation_id, s.conversation_id, ss.conversation_id, msg.conversation_id) AS conversation_id,
    COALESCE(m.created_at, s.created_at, ss.created_at, d.created_at) AS created_at
FROM
    unique_results u
    LEFT JOIN messages m ON u.source_id = m.id
        AND u.source_type = 'message'
    LEFT JOIN message_contents mc ON m.id = mc.message_id
        AND u.source_type = 'message'
    LEFT JOIN summaries s ON u.source_id = s.id
        AND u.source_type = 'summary'
    LEFT JOIN search_topic_syntheses ss ON u.source_id = ss.id
        AND u.source_type = 'search'
    LEFT JOIN documents d ON u.source_id = d.id
        AND u.source_type = 'document'
    LEFT JOIN messages msg ON d.message_id = msg.id
        AND u.source_type = 'document'
WHERE
    u.pair_key IN (
        SELECT
            pair_key
        FROM
            limited_pairs)
ORDER BY
    u.similarity DESC, -- Sort by highest similarity first
    COALESCE(m.conversation_id, s.conversation_id, ss.conversation_id, msg.conversation_id), -- Keep conversation pairs together
    COALESCE(m.created_at, s.created_at, ss.created_at, d.created_at) -- Maintain chronological order within conversations
