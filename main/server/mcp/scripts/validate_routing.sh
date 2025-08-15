#!/bin/bash

set -e

echo "Validating routing at 08:05 AM EDT..."
endpoints=("troubleshoot" "auth/oauth" "health" "404")
for endpoint in "${endpoints[@]}"; do
  response=$(curl -s -o /dev/null -w "%{http_code}" https://webxos.netlify.app/api/$endpoint)
  if [ $response -ne 200 ]; then
    echo "❌ $endpoint returned $response"
    exit 1
  fi
  echo "✅ $endpoint validated"
done
echo "Routing validation passed"
