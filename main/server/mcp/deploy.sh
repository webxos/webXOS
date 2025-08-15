#!/bin/bash
echo "Starting deployment..."
npm install node-fetch
pip install flask
# Verify function files exist
for func in auth/oauth troubleshoot health 404; do
  if [ ! -f "netlify/functions/api/$func.js" ]; then
    echo "Error: netlify/functions/api/$func.js not found"
    exit 1
  fi
done
# Verify Netlify CLI
if ! command -v netlify &> /dev/null; then
    echo "Error: Netlify CLI not installed"
    exit 1
fi
netlify deploy --prod --dir=main --debug
echo "Validating endpoints..."
./server/mcp/scripts/validate_routing.sh
echo "Deployment complete."
