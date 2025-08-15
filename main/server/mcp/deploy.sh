#!/bin/bash

set -e

echo "🚀 Starting deployment validation..."

echo "Building project..."
npm run build

echo "Verifying function deployment..."
if [ ! -d "netlify/functions" ]; then
  echo "❌ Functions directory missing"
  exit 1
fi

echo "Testing endpoints..."
node scripts/health-check.js || {
  echo "❌ Health check failed"
  exit 1
}

echo "Deploying to Netlify..."
netlify deploy --prod --dir=main/app

echo "Post-deployment validation..."
curl -s -X POST -H "Content-Type: application/json" -d '{"provider":"mock","code":"test_code"}' https://webxos.netlify.app/api/auth/oauth -o /dev/null -w "%{http_code}" | grep -q 200 || {
  echo "❌ OAuth endpoint failed"
  exit 1
}
curl -s -X POST https://webxos.netlify.app/api/troubleshoot -o /dev/null -w "%{http_code}" | grep -q 200 || {
  echo "❌ Troubleshoot endpoint failed"
  exit 1
}

echo "✅ Deployment completed successfully at $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
