-- Add analysis to database using IntentAnalysis schema
INSERT INTO analyses(message_id, workflow_type, complexity_level, required_capabilities, domain_specificity, reusability_potential, confidence, response_format, technical_domain, requires_tools, requires_custom_tools, tool_complexity_score, computational_requirements, created_at)
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, COALESCE($14, NOW()))
RETURNING
    id;

