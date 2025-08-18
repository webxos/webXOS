#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in STACK_AUTH_CLIENT_ID STACK_AUTH_CLIENT_SECRET NEON_PROJECT_ID DATABASE_URL JWT_AUDIENCE; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [auth_setup.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Verify Stack Auth configuration
echo "Verifying Stack Auth configuration..."
curl -s -o /dev/null -w "%{http_code}" "https://api.stack-auth.com/api/v1/projects/142ad169-5d57-4be3-bf41-6f3cd0a9ae1d/.well-known/jwks.json" | grep -q "200" || {
  echo "Error: JWKS URL not accessible [auth_setup.sh:15] [ID:jwks_access_error]"
  exit 1
}

# Create wallet_transactions table
echo "Creating wallet_transactions table..."
PGPASSWORD=$(echo $DATABASE_URL | sed -E 's/.*:(.*)@.*/\1/') psql "$DATABASE_URL" -c "
CREATE TABLE IF NOT EXISTS wallet_transactions (
    transaction_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,
    amount FLOAT,
    data JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    project_id TEXT NOT NULL
);
GRANT SELECT, INSERT, UPDATE ON wallet_transactions TO replication_user;
ALTER TABLE wallet_transactions ENABLE ROW LEVEL SECURITY;
CREATE POLICY wallet_access ON wallet_transactions
    USING (project_id = '$NEON_PROJECT_ID' AND user_id = current_user);
"

# Test OAuth2 authorize URL
echo "Testing OAuth2 authorize URL..."
curl -s -o /dev/null -w "%{http_code}" "https://api.stack-auth.com/api/v1/oauth/authorize?client_id=$STACK_AUTH_CLIENT_ID&redirect_uri=https://webxos.netlify.app/vial2.html&response_type=code&scope=openid+profile+email" | grep -q "200" || {
  echo "Error: OAuth2 authorize URL not accessible [auth_setup.sh:20] [ID:oauth_access_error]"
  exit 1
}

echo "Auth setup completed [auth_setup.sh:25] [ID:auth_setup_success]"
