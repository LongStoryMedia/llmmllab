-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    content TEXT NOT NULL, -- base64 encoded for binary, plain text for text files
    text_content TEXT, -- extracted text content for searchability (null for images)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_documents_conversation_id ON documents(conversation_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_content_type ON documents(content_type);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);

-- Create hypertable for documents with optimal chunk interval
SELECT create_hypertable('documents', 'created_at', 
    if_not_exists => TRUE, 
    migrate_data => TRUE, 
    chunk_time_interval => interval '30 days');