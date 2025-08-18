#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in DATABASE_URL NEON_PROJECT_ID; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [relay_signal_setup.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Apply user sessions migration
echo "Applying user sessions migration..."
./scripts/migrate.sh || {
  echo "Error: Migration failed [relay_signal_setup.sh:15] [ID:migration_error]"
  exit 1
}

# Verify database connectivity
echo "Verifying database connectivity..."
PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -c "SELECT 1;" > /dev/null || {
  echo "Error: Database connectivity failed [relay_signal_setup.sh:20] [ID:db_connect_error]"
  exit 1
}

echo "Relay signal setup completed [relay_signal_setup.sh:25] [ID:relay_signal_setup_success]"
