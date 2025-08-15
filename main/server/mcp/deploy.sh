#!/bin/bash
echo "Starting deployment..."
npm install node-fetch
pip install flask
# Verify function files exist
for func in oauth troubleshoot health 404; do
  if [ ! -f "netlify/functions/api/$func.js" ]; then
    echo "Error: netlify/functions/api/$func.js not found"
    exit 1
  fi
done
netlify deploy --prod --dir=main
echo "Validating endpoints..."
./server/mcp/scripts/validate_routing.sh
echo "Deployment complete."
