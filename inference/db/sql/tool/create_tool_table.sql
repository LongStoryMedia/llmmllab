-- Create the dynamic_tools table with LangChain BaseTool interface support
CREATE TABLE IF NOT EXISTS dynamic_tools(
    id serial PRIMARY KEY,
    user_id text NOT NULL,
    name text NOT NULL,
    description text NOT NULL,
    code text NOT NULL,
    function_name text NOT NULL,
    embedding vector(768), -- Add embedding vector for semantic search (768 dimensions)
    -- LangChain BaseTool interface fields
    args_schema jsonb, -- JSON schema dict, Pydantic model reference, or null
    return_direct boolean DEFAULT FALSE,
    tags text[], -- Array of tag strings
    metadata jsonb DEFAULT '{}', -- Flexible metadata object
    handle_tool_error text, -- Can store boolean, string, or null (serialized)
    handle_validation_error text, -- Can store boolean, string, or null (serialized)
    response_format text DEFAULT 'content' CHECK (response_format IN ('content', 'content_and_artifact')),
    -- Legacy field (kept for backward compatibility)
    parameters jsonb,
    -- Timestamps
    created_at timestamp with time zone DEFAULT NOW(),
    updated_at timestamp with time zone DEFAULT NOW(),
    CONSTRAINT fk_user_id FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create index on user_id for faster lookup
CREATE INDEX IF NOT EXISTS idx_dynamic_tools_user_id ON dynamic_tools(user_id);

-- Create trigger to update timestamp on updates
CREATE OR REPLACE FUNCTION update_dynamic_tools_timestamp()
    RETURNS TRIGGER
    AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$
LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trigger_update_dynamic_tools_timestamp
    BEFORE UPDATE ON dynamic_tools
    FOR EACH ROW
    EXECUTE FUNCTION update_dynamic_tools_timestamp();

