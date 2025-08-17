#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in DATABASE_URL NEON_PROJECT_ID; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [migrate.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Apply migrations
echo "Applying migrations..."
python -m vial2.mcp.api.migrations.migrate

# Create playing_with_neon table for replication testing
echo "Creating playing_with_neon table..."
PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -c "CREATE TABLE IF NOT EXISTS playing_with_neon(id SERIAL PRIMARY KEY, name TEXT NOT NULL, value REAL);"
PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -c "INSERT INTO playing_with_neon(name, value) SELECT LEFT(md5(i::TEXT), 10), random() FROM generate_series(1, 10) s(i);"

# Verify table creation
row_count=$(PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM playing_with_neon;")
if [ "$row_count" -ge 10 ]; then
  echo "playing_with_neon table created successfully with $row_count rows"
else
  echo "Error: Failed to create playing_with_neon table [migrate.sh:25] [ID:table_create_error]"
  exit 1
fi
