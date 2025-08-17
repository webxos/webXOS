#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in DATABASE_URL NEON_PROJECT_ID NETLIFY_AUTH_TOKEN NETLIFY_SITE_ID; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [build.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Install dependencies
pip install -r vial2/mcp/api/requirements.txt

# Copy vial2.html to root for Netlify
cp vial2.html .

# Verify file copy
if [ ! -f "vial2.html" ]; then
  echo "Error: Failed to copy vial2.html to root [build.sh:20] [ID:file_error]"
  exit 1
fi

# Build static assets (CSS/JS)
mkdir -p dist/static/css dist/static/js
cp -r vial2/static/css/* dist/static/css/
cp -r vial2/static/js/* dist/static/js/

echo "Build completed successfully"
