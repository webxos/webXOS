#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in DATABASE_URL NEON_PROJECT_ID; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [monitor_replication.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Query replication status
echo "Checking replication status..."
PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -c "SELECT subname, received_lsn, latest_end_lsn, last_msg_receipt_time FROM pg_stat_subscription;" > replication_status.log

# Check for active subscriptions
if grep -q "sub_" replication_status.log; then
  echo "Active subscriptions found"
else
  echo "Error: No active subscriptions found [monitor_replication.sh:20] [ID:no_subscriptions_error]"
  exit 1
fi

# Log row count for playing_with_neon
row_count=$(PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM playing_with_neon;")
echo "playing_with_neon row count: $row_count"
if [ "$row_count" -lt 10 ]; then
  echo "Error: playing_with_neon row count too low: $row_count [monitor_replication.sh:25] [ID:row_count_error]"
  exit 1
fi
