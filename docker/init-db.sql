-- Initialize DeepSea AI database
CREATE DATABASE IF NOT EXISTS deepsea_ai;

-- Create user and grant privileges
CREATE USER IF NOT EXISTS deepsea WITH PASSWORD 'deepsea123';
GRANT ALL PRIVILEGES ON DATABASE deepsea_ai TO deepsea;

-- Connect to the database
\c deepsea_ai;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO deepsea;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO deepsea;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO deepsea;