CREATE TABLE IF NOT EXISTS search_queries(
  id serial PRIMARY KEY,
  query text NOT NULL,
  created_at timestamp DEFAULT CURRENT_TIMESTAMP
);

