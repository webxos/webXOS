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

echo "Validating routes..."
python server/mcp/config/route_validator.py || {
  echo "Error: Route validation failed. Check netlify.toml routing."
  exit 1
}

echo "Checking OAuth configuration..."
if [ ! -f "server/mcp/config/oauth_config.py" ]; then
  echo "Warning: oauth_config.py is missing. Ensure OAuth provider credentials are configured."
fi

echo "Running health check..."
curl -s http://localhost:8081/health | grep -q "healthy" || {
  echo "Error: Health check failed. Server may be unhealthy."
  exit 1
}

echo "Deployment completed successfully at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
