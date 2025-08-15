#!/bin/bash
# main/server/scripts/setup_db.sh

set -e

# MongoDB setup
MONGODB_URI=${MONGODB_URI:-"mongodb://localhost:27017"}
DB_NAME="vial_mcp"

echo "Setting up MongoDB database: $DB_NAME"
mongosh $MONGODB_URI <<EOF
use $DB_NAME
db.createCollection("notes")
db.notes.createIndex({"user_id": 1})
db.notes.createIndex({"tags": 1})
db.notes.createIndex({"content": "text"})
db.createCollection("wallets")
db.wallets.createIndex({"address": 1}, {"unique": true})
EOF

# Redis setup
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-"6379"}

echo "Setting up Redis"
redis-cli -h $REDIS_HOST -p $REDIS_PORT <<EOF
FLUSHALL
CONFIG SET maxmemory 512mb
CONFIG SET maxmemory-policy allkeys-lru
EOF

echo "Database setup complete"
