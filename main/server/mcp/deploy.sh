#!/bin/bash

set -e

echo "Building project..."
npm run build

echo "Deploying to Netlify..."
netlify deploy --prod --dir=main/app

echo "Verifying function deployment..."
netlify functions:list | grep -E "auth|troubleshoot" > /dev/null 2>&1 || {
  echo "Error: Functions not deployed correctly. Check Netlify configuration."
  exit 1
}

echo "Checking endpoint availability..."
curl -s -o /dev/null -w "%{http_code}" https://webxos.netlify.app/vial2/api/troubleshoot | grep -q 200 || {
  echo "Error: Troubleshoot endpoint not found (HTTP 404). Verify routing in netlify.toml."
  exit 1
}
curl -s -o /dev/null -w "%{http_code}" https://webxos.netlify.app/vial2/api/auth/oauth | grep -q 200 || {
  echo "Error: OAuth endpoint not found (HTTP 404). Verify routing in netlify.toml."
  exit 1
}

echo "Running health check..."
curl -s http://localhost:8081/health | grep -q "healthy" || {
  echo "Error: Health check failed. Server may be unhealthy."
  exit 1
}

echo "Deployment completed successfully at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
