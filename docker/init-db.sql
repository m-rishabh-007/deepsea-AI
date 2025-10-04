-- Create database and role only if they do not already exist
SELECT 'CREATE DATABASE deepsea_ai'
WHERE NOT EXISTS (
	SELECT FROM pg_database WHERE datname = 'deepsea_ai'
)\gexec

SELECT 'CREATE ROLE deepsea WITH LOGIN PASSWORD ''deepsea123''' 
WHERE NOT EXISTS (
	SELECT FROM pg_roles WHERE rolname = 'deepsea'
)\gexec

GRANT ALL PRIVILEGES ON DATABASE deepsea_ai TO deepsea;

-- Connect to the database
\c deepsea_ai;

-- Grant schema privileges
GRANT ALL ON SCHEMA public TO deepsea;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO deepsea;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO deepsea;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO deepsea;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO deepsea;