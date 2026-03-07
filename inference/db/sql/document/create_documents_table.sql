-- Create documents table with TimescaleDB compatible schema
CREATE TABLE IF NOT EXISTS documents(
    id serial,
    message_id integer NOT NULL,
    user_id text NOT NULL,
    filename text NOT NULL,
    content_type text NOT NULL,
    file_size integer NOT NULL,
    content text NOT NULL, -- base64 encoded for binary, plain text for text files
    text_content text, -- extracted text content for searchability (null for images)
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at) -- Composite primary key required for TimescaleDB
    -- Note: Foreign key constraints removed for TimescaleDB compatibility
    -- Application layer handles referential integrity
    -- FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE,
    -- FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for efficient queries (must include created_at for unique indexes)
CREATE INDEX IF NOT EXISTS idx_documents_message_id ON documents(message_id, created_at);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id, created_at);

CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

-- Note: Primary key already provides index for (id, created_at)
