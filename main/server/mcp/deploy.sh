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

echo "Testing OAuth endpoint..."
curl -s -X POST -H "Content-Type: application/json" -d '{"provider":"mock","code":"test_code"}' https://webxos.netlify.app/vial2/api/auth/oauth -o /dev/null -w "%{http_code}" | grep -q 200 || {
  echo "Error: OAuth endpoint returned non-200 status. Check routing or deployment."
  exit 1
}

echo "Testing Troubleshoot endpoint..."
curl -s -X POST https://webxos.netlify.app/vial2/api/troubleshoot -o /dev/null -w "%{http_code}" | grep -q 200 || {
  echo "Error: Troubleshoot endpoint returned non-200 status. Check routing or deployment."
  exit 1
}

echo "Running health check..."
curl -s http://localhost:8081/health | grep -q "healthy" || {
  echo "Error: Health check failed. Server may be unhealthy."
  exit 1
}

echo "Deployment completed successfully at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
