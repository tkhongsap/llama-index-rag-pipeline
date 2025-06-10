-- Initialize iLand test database with PGVector extension
-- This script runs when the PostgreSQL container starts for the first time

-- Enable PGVector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create iland_md_data table for storing processed documents
CREATE TABLE IF NOT EXISTS iland_md_data (
    id SERIAL PRIMARY KEY,
    deed_id TEXT NOT NULL,
    md_string TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on deed_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_iland_md_data_deed_id ON iland_md_data (deed_id);

-- Create iland_embeddings table for storing vectors (will be created by LlamaIndex, but preparing structure)
-- Note: The actual table will be created by PGVectorStore with proper vector dimensions

-- Create test user permissions (if needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO iland_test_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO iland_test_user;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'iLand test database initialized successfully';
    RAISE NOTICE 'Tables created: iland_md_data';
    RAISE NOTICE 'Extensions enabled: vector';
END $$; 