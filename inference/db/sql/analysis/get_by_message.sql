-- Get analyses by message ID with IntentAnalysis schema
SELECT
    id,
    message_id,
    workflow_type,
    complexity_level,
    required_capabilities,
    domain_specificity,
    reusability_potential,
    confidence,
    response_format,
    technical_domain,
    requires_tools,
    requires_custom_tools,
    tool_complexity_score,
    computational_requirements,
    created_at
FROM
    analyses
WHERE
    message_id = $1
ORDER BY
    created_at ASC;

