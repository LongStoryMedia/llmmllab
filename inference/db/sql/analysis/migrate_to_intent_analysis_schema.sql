-- Simple migration for analyses table - handles existing tables gracefully
-- Avoids DO blocks that might cause syntax issues

-- Step 1: Add missing columns (will be ignored if they already exist)
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS workflow_type TEXT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS complexity_level TEXT;  
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS required_capabilities JSONB;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS domain_specificity NUMERIC(3, 2);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS reusability_potential NUMERIC(3, 2);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS confidence NUMERIC(3, 2);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS response_format TEXT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS technical_domain TEXT;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS requires_tools BOOLEAN DEFAULT FALSE;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS requires_custom_tools BOOLEAN DEFAULT FALSE;
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS tool_complexity_score NUMERIC(3, 2);
ALTER TABLE analyses ADD COLUMN IF NOT EXISTS computational_requirements JSONB;

-- Step 2: Update NULL values with defaults from analysis_data JSONB column
UPDATE analyses 
SET 
    workflow_type = COALESCE(workflow_type, analysis_data->>'workflow_type', 'unknown'),
    complexity_level = COALESCE(complexity_level, analysis_data->>'complexity_level', 'unknown'),
    required_capabilities = COALESCE(required_capabilities, analysis_data->'required_capabilities', '[]'::jsonb),
    domain_specificity = COALESCE(domain_specificity, (analysis_data->>'domain_specificity')::numeric, 0.0),
    reusability_potential = COALESCE(reusability_potential, (analysis_data->>'reusability_potential')::numeric, 0.0),
    confidence = COALESCE(confidence, (analysis_data->>'confidence')::numeric, 0.0),
    response_format = COALESCE(response_format, analysis_data->>'response_format'),
    technical_domain = COALESCE(technical_domain, analysis_data->>'technical_domain'),
    requires_tools = COALESCE(requires_tools, (analysis_data->>'requires_tools')::boolean, false),
    requires_custom_tools = COALESCE(requires_custom_tools, (analysis_data->>'requires_custom_tools')::boolean, false),
    tool_complexity_score = COALESCE(tool_complexity_score, (analysis_data->>'tool_complexity_score')::numeric, 0.0),
    computational_requirements = COALESCE(computational_requirements, analysis_data->'computational_requirements', '{}'::jsonb)
WHERE workflow_type IS NULL OR complexity_level IS NULL;