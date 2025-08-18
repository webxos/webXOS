#!/bin/bash

set -e

# Check for wallet data file
if [ ! -f "wallet.md" ]; then
  echo "Error: wallet.md not found [env_setup.sh:10] [ID:wallet_file_error]"
  exit 1
fi

# Extract wallet address (simulated parsing, replace with actual .md parsing logic)
WALLET_ADDRESS=$(grep -oP 'address: \K0x[a-fA-F0-9]{40}' wallet.md || echo "")
if [ -z "$WALLET_ADDRESS" ]; then
  echo "Error: Invalid or missing wallet address in wallet.md [env_setup.sh:15] [ID:wallet_address_error]"
  exit 1
fi

# Generate .env file
cat > .env << EOF
DATABASE_URL=postgresql://neondb_owner:npg_EzPpBWkGdm69@ep-sparkling-thunder-aetjtveu-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require
DATA_API_URL=https://app-billowing-king-08029676.dpl.myneon.app
STACK_AUTH_CLIENT_ID=your_stack_auth_client_id
STACK_AUTH_CLIENT_SECRET=your_stack_auth_client_secret
NEON_API_KEY=your_neon_api_key
NEON_PROJECT_ID=twilight-art-21036984
JWT_SECRET_KEY=your_jwt_secret_key
NETLIFY_AUTH_TOKEN=your_netlify_auth_token
NETLIFY_SITE_ID=your_netlify_site_id
GITHUB_TOKEN=your_github_token
GITHUB_REPOSITORY=your_repo_owner/your_repo_name
JWT_AUDIENCE=vial-mcp-webxos
SOURCE_DB_HOST=your_source_host
SOURCE_DB_NAME=postgres
SOURCE_DB_USER=replication_user
SOURCE_DB_PASSWORD=your_secure_password
WALLET_ADDRESS=$WALLET_ADDRESS
EOF

# Verify .env creation
if [ -f ".env" ]; then
  echo "Generated .env with wallet address: $WALLET_ADDRESS [env_setup.sh:20] [ID:env_setup_success]"
else
  echo "Error: Failed to generate .env [env_setup.sh:25] [ID:env_setup_error]"
  exit 1
fi
