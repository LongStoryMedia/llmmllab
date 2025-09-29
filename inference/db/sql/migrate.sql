INSERT INTO conversations(id, user_id, title, created_at, updated_at)
SELECT
    rd.id,
    rd.user_id,
    rd.title,
    rd.created_at,
    rd.updated_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, user_id, title, created_at, updated_at FROM conversations') AS rd(id integer,
        user_id text,
        title text,
        created_at timestamptz,
        updated_at timestamptz);

INSERT INTO messages(id, conversation_id, role, created_at)
SELECT
    rd.id,
    rd.conversation_id,
    rd.role,
    rd.created_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, conversation_id, role, created_at FROM messages') AS rd(id integer,
        conversation_id integer,
        ROLE TEXT,
        created_at timestamptz);

INSERT INTO message_contents(message_id, created_at, type, text_content, url)
SELECT
    rd.id,
    rd.created_at,
    'text' AS type, -- Default type for text content
    rd.content,
    NULL AS url -- No URL for text content
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, created_at, content FROM messages') AS rd(id integer,
        created_at timestamptz,
        content text);

INSERT INTO summaries(id, conversation_id, content, level, source_ids, created_at)
SELECT
    rd.id,
    rd.conversation_id,
    rd.content,
    rd.level,
    rd.source_ids,
    rd.created_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, conversation_id, content, level, source_ids, created_at FROM summaries') AS rd(id integer,
        conversation_id integer,
        content text,
        level integer,
        source_ids text[],
        created_at timestamptz);

INSERT INTO memories(id, user_id, source_id, source, role, embedding, created_at)
SELECT
    rd.id,
    rd.user_id,
    rd.source_id,
    rd.source,
    rd.role,
    rd.embedding,
    rd.created_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, user_id, source_id, source, role, embedding, created_at FROM memories') AS rd(id integer,
        user_id text,
        source_id integer,
        source text,
        ROLE text,
        embedding vector(768),
        created_at timestamptz);

INSERT INTO images(id, filename, thumbnail, format, width, height, conversation_id, user_id, created_at)
SELECT
    rd.id,
    rd.filename,
    rd.thumbnail,
    rd.format,
    rd.width,
    rd.height,
    rd.conversation_id,
    rd.user_id,
    rd.created_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, filename, thumbnail, format, width, height, conversation_id, user_id, created_at FROM images') AS rd(id integer,
        filename text,
        thumbnail text,
        format text,
        width integer,
        height integer,
        conversation_id integer,
        user_id text,
        created_at timestamptz);

INSERT INTO model_profiles(id, user_id, name, description, model_name, parameters, system_prompt, model_version, type, created_at, updated_at)
SELECT
    rd.id,
    rd.user_id,
    rd.name,
    rd.description,
    rd.model_name,
    rd.parameters,
    rd.system_prompt,
    rd.model_version,
    rd.type,
    rd.created_at,
    rd.updated_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, user_id, name, description, model_name, parameters, system_prompt, model_version, type, created_at, updated_at FROM model_profiles') AS rd(id uuid,
        user_id text,
        name text,
        description text,
        model_name text,
        parameters jsonb,
        system_prompt text,
        model_version text,
        type integer,
        created_at timestamptz,
        updated_at timestamptz);

INSERT INTO research_tasks(id, user_id, conversation_id, query, status, result, created_at, updated_at)
SELECT
    rd.id,
    rd.user_id,
    rd.conversation_id,
    rd.query,
    rd.status,
    rd.result,
    rd.created_at,
    rd.updated_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, user_id, conversation_id, query, status, result, created_at, updated_at FROM research_tasks') AS rd(id uuid,
        user_id text,
        conversation_id text,
        query text,
        status text,
        result text,
        created_at timestamptz,
        updated_at timestamptz);

INSERT INTO research_subtasks(id, task_id, title, description, status, result, created_at, updated_at)
SELECT
    rd.id,
    rd.task_id,
    rd.title,
    rd.description,
    rd.status,
    rd.result,
    rd.created_at,
    rd.updated_at
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, task_id, title, description, status, result, created_at, updated_at FROM research_subtasks') AS rd(id integer,
        task_id integer,
        title text,
        description text,
        status text,
        result text,
        created_at timestamptz,
        updated_at timestamptz);

INSERT INTO users(id, username, created_at, config)
SELECT
    rd.id,
    rd.username,
    rd.created_at,
    rd.config
FROM
    dblink('host=localhost port=5432 dbname=ollama user=lsm password=7cb9c812e384e16c911a72f1066517d205e8641b78edb3b1b3c78d0c351b1885', 'SELECT id, username, created_at, config FROM users') AS rd(id integer,
        username text,
        created_at timestamptz,
        config jsonb);

