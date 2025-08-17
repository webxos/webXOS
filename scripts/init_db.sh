#!/bin/bash

set -e

# Load environment variables
source .env

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
  echo "Error: DATABASE_URL not set [init_db.sh:10] [ID:env_error]"
  exit 1
fi

# Apply schema to Neon database
psql -d "$DATABASE_URL" -f vial2/mcp/api/database/schema.sql

# Verify schema application
psql -d "$DATABASE_URL" -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'" > /tmp/schema_check.txt
if grep -q "users" /tmp/schema_check.txt && grep -q "vials" /tmp/schema_check.txt; then
  echo "Database schema initialized successfully"
else
  echo "Error: Schema initialization failed [init_db.sh:20] [ID:schema_error]"
  exit 1
fi
