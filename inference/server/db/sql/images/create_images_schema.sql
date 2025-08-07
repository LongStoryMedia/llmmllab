CREATE TABLE IF NOT EXISTS images(
    id serial,
    filename text NOT NULL,
    thumbnail text NOT NULL,
    format text NOT NULL,
    width integer NOT NULL,
    height integer NOT NULL,
    conversation_id integer NOT NULL,
    user_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    -- FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    PRIMARY KEY (id, created_at)
);

CREATE INDEX IF NOT EXISTS idx_images_conversation_id ON images(conversation_id);

CREATE INDEX IF NOT EXISTS idx_images_user_id ON images(user_id);

-- Create hypertable for images with optimal chunk interval
SELECT
    create_hypertable('images', 'created_at', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '7 days');

