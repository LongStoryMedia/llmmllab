-- Initialize database for integration testing
-- This script is run automatically when the PostgreSQL container is first created

-- Create extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- Configure TimescaleDB
SELECT set_config('timescaledb.telemetry_enabled', 'off', false);

-- Create initial schema (will be recreated by app)
-- This is just a placeholder to ensure the database is ready

-- Create a test user (if needed)
-- CREATE USER test_user WITH PASSWORD 'test_password';
-- GRANT ALL PRIVILEGES ON DATABASE llmmll_test TO test_user;