#!/bin/bash
endpoints=(
  "https://webxos.netlify.app/.netlify/functions/oauth"
  "https://webxos.netlify.app/.netlify/functions/troubleshoot"
  "https://webxos.netlify.app/.netlify/functions/health"
  "https://webxos.netlify.app/.netlify/functions/404"
)
for endpoint in "${endpoints[@]}"; do
  status=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint")
  if [ "$status" -eq 200 ]; then
    echo "Endpoint $endpoint is reachable."
  else
    echo "Error: Endpoint $endpoint returned $status."
    curl -s "$endpoint" | head -n 10
    exit 1
  fi
done
echo "All endpoints validated successfully."
