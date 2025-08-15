#!/bin/bash
echo "Starting deployment..."
npm install node-fetch
pip install flask
# Verify Netlify CLI
if ! command -v netlify &> /dev/null; then
    echo "Error: Netlify CLI not installed"
    exit 1
fi
# Verify function files exist
for func in oauth troubleshoot health 404; do
  if [ ! -f "netlify/functions/$func.js" ]; then
    echo "Error: netlify/functions/$func.js not found"
    exit 1
  fi
done
# Deploy with debug output
netlify deploy --prod --dir=main --debug
# Wait for deployment to complete
sleep 5
echo "Validating endpoints..."
./server/mcp/scripts/validate_routing.sh
echo "Deployment complete."
