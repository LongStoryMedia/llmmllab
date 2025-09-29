-- Create search_topic_syntheses table if not exists
CREATE TABLE IF NOT EXISTS search_topic_syntheses(
    id serial,
    urls jsonb NOT NULL,
    topics jsonb NOT NULL,
    synthesis text NOT NULL,
    created_at timestamp with time zone DEFAULT NOW(),
    conversation_id integer,
    PRIMARY KEY (id, created_at)
);

-- Create index for faster searches
CREATE INDEX IF NOT EXISTS idx_search_topic_syntheses_conversation_id ON search_topic_syntheses(conversation_id);

CREATE INDEX IF NOT EXISTS idx_search_topic_syntheses_created_at ON search_topic_syntheses(created_at);

