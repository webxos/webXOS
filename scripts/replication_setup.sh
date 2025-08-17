#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in SOURCE_DB_HOST SOURCE_DB_NAME SOURCE_DB_USER SOURCE_DB_PASSWORD NEON_DATABASE_URL NEON_PROJECT_ID; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [replication_setup.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Configure source database for logical replication
echo "Configuring source database for logical replication..."
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "ALTER SYSTEM SET wal_level = logical;"
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "CREATE ROLE replication_user WITH REPLICATION LOGIN PASSWORD '$SOURCE_DB_PASSWORD';"
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "GRANT USAGE ON SCHEMA public TO replication_user;"
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO replication_user;"
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO replication_user;"
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "CREATE PUBLICATION my_publication FOR TABLE playing_with_neon;"

# Restart source database (assuming local or accessible control)
# Note: For hosted providers, follow their specific restart procedure
echo "Reloading source database configuration..."
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "SELECT pg_reload_conf();"

# Verify wal_level
wal_level=$(PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -t -c "SHOW wal_level;")
if [ "$wal_level" = "logical" ]; then
  echo "Logical replication enabled successfully"
else
  echo "Error: Failed to set wal_level to logical [replication_setup.sh:25] [ID:wal_level_error]"
  exit 1
fi

# Create sample table if not exists
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "CREATE TABLE IF NOT EXISTS playing_with_neon(id SERIAL PRIMARY KEY, name TEXT NOT NULL, value REAL);"
PGPASSWORD="$SOURCE_DB_PASSWORD" psql -h "$SOURCE_DB_HOST" -U "$SOURCE_DB_USER" -d "$SOURCE_DB_NAME" -c "INSERT INTO playing_with_neon(name, value) SELECT LEFT(md5(i::TEXT), 10), random() FROM generate_series(1, 10) s(i);"
