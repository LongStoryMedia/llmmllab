-- Migration to update analyses table schema to match IntentAnalysis model
-- This migration extracts data from the generic analysis_data JSONB column into specific columns

-- Step 1: Add new columns to match IntentAnalysis fields
ALTER TABLE analyses 
ADD COLUMN IF NOT EXISTS workflow_type TEXT,
ADD COLUMN IF NOT EXISTS complexity_level TEXT,
ADD COLUMN IF NOT EXISTS required_capabilities JSONB,
ADD COLUMN IF NOT EXISTS domain_specificity NUMERIC(3, 2) CHECK (domain_specificity >= 0.0 AND domain_specificity <= 1.0),
ADD COLUMN IF NOT EXISTS reusability_potential NUMERIC(3, 2) CHECK (reusability_potential >= 0.0 AND reusability_potential <= 1.0),
ADD COLUMN IF NOT EXISTS confidence NUMERIC(3, 2) CHECK (confidence >= 0.0 AND confidence <= 1.0),
ADD COLUMN IF NOT EXISTS response_format TEXT,
ADD COLUMN IF NOT EXISTS technical_domain TEXT,
ADD COLUMN IF NOT EXISTS requires_tools BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS requires_custom_tools BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS tool_complexity_score NUMERIC(3, 2) CHECK (tool_complexity_score >= 0.0 AND tool_complexity_score <= 1.0),
ADD COLUMN IF NOT EXISTS computational_requirements JSONB;

-- Step 2: Migrate existing data from analysis_data JSONB to new columns
-- Only run this if there are existing rows and the new columns are empty
UPDATE analyses 
SET 
    workflow_type = COALESCE(analysis_data->>'workflow_type', 'unknown'),
    complexity_level = COALESCE(analysis_data->>'complexity_level', 'unknown'),
    required_capabilities = COALESCE(analysis_data->'required_capabilities', '[]'::jsonb),
    domain_specificity = COALESCE((analysis_data->>'domain_specificity')::numeric, 0.0),
    reusability_potential = COALESCE((analysis_data->>'reusability_potential')::numeric, 0.0),
    confidence = COALESCE((analysis_data->>'confidence')::numeric, 0.0),
    response_format = analysis_data->>'response_format',
    technical_domain = analysis_data->>'technical_domain',
    requires_tools = COALESCE((analysis_data->>'requires_tools')::boolean, false),
    requires_custom_tools = COALESCE((analysis_data->>'requires_custom_tools')::boolean, false),
    tool_complexity_score = COALESCE((analysis_data->>'tool_complexity_score')::numeric, 0.0),
    computational_requirements = COALESCE(analysis_data->'computational_requirements', '{}'::jsonb)
WHERE workflow_type IS NULL;

-- Step 3: Make required columns NOT NULL after data migration
ALTER TABLE analyses 
ALTER COLUMN workflow_type SET NOT NULL,
ALTER COLUMN complexity_level SET NOT NULL,
ALTER COLUMN required_capabilities SET NOT NULL,
ALTER COLUMN domain_specificity SET NOT NULL,
ALTER COLUMN reusability_potential SET NOT NULL,
ALTER COLUMN confidence SET NOT NULL,
ALTER COLUMN requires_tools SET NOT NULL,
ALTER COLUMN requires_custom_tools SET NOT NULL,
ALTER COLUMN tool_complexity_score SET NOT NULL,
ALTER COLUMN computational_requirements SET NOT NULL;

-- Step 4: Create new indexes for the structured columns
CREATE INDEX IF NOT EXISTS idx_analyses_workflow_type ON analyses(workflow_type);
CREATE INDEX IF NOT EXISTS idx_analyses_complexity_level ON analyses(complexity_level);
CREATE INDEX IF NOT EXISTS idx_analyses_technical_domain ON analyses(technical_domain) WHERE technical_domain IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_analyses_requires_tools ON analyses(requires_tools);
CREATE INDEX IF NOT EXISTS idx_analyses_requires_custom_tools ON analyses(requires_custom_tools);
CREATE INDEX IF NOT EXISTS idx_analyses_confidence ON analyses(confidence);
CREATE INDEX IF NOT EXISTS idx_analyses_required_capabilities ON analyses USING GIN (required_capabilities);
CREATE INDEX IF NOT EXISTS idx_analyses_computational_requirements ON analyses USING GIN (computational_requirements);

-- Step 5: Drop the old generic analysis_data column (optional - keep for backwards compatibility)
-- ALTER TABLE analyses DROP COLUMN IF EXISTS analysis_data;

-- Note: We keep analysis_data column for now to maintain backwards compatibility
-- It can be dropped in a future migration once all code is updated