-- Returns only “active” summaries for a conversation.
-- A summary is considered compressed (and therefore excluded) if its id appears
-- in the source_ids array of any higher-level summary for the same conversation.
-- This yields the set of leaf (not yet compressed) summaries plus the highest-level summaries.
WITH referencing AS (
  -- Expand jsonb array source_ids into scalar elements.
  -- Assumes each element stores the UUID (or text identifier) of a lower-level summary.
  SELECT DISTINCT
    (elem)::int AS ref_id
  FROM
    summaries s2
    CROSS JOIN LATERAL jsonb_array_elements_text(s2.source_ids) AS elem
  WHERE
    s2.conversation_id = $1
    AND s2.source_ids IS NOT NULL
)
SELECT
  s.id,
  s.conversation_id,
  s.content,
  s.level,
  s.source_ids,
  s.created_at
FROM
  summaries s
WHERE
  s.conversation_id = $1
  AND NOT EXISTS (
    SELECT
      1
    FROM
      referencing r
    WHERE
      r.ref_id = s.id)
ORDER BY
  s.level ASC,
  s.created_at DESC
