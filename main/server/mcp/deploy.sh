#!/bin/bash

# Exit on any error
set -e

# Build the project
echo "Building project..."
npm run build

# Deploy to Netlify
echo "Deploying to Netlify..."
netlify deploy --prod --dir=main/app

# Verify function deployment
echo "Verifying function deployment..."
netlify functions:list | grep -E "auth|troubleshoot" > /dev/null 2>&1 || {
  echo "Error: Functions not deployed correctly. Check Netlify configuration."
  exit 1
}

echo "Deployment completed successfully at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
