#!/bin/bash

LOG_FILE="db/errorlog.md"
PORTS=(8000 8001 8002 8003 8004 8005 8006 8007)

echo "Checking Vial MCP services..."

for port in "${PORTS[@]}"; do
    curl -s -m 5 "http://localhost:$port/api/health" > /tmp/health_check.json
    if [ $? -eq 0 ]; then
        status=$(jq -r '.status' /tmp/health_check.json)
        echo "## [$(date -u +%Y-%m-%dT%H:%M:%SZ)] INFO: Service on port $port is $status" >> "$LOG_FILE"
        echo "Port $port: $status"
    else
        echo "## [$(date -u +%Y-%m-%dT%H:%M:%SZ)] ERROR: Service on port $port unreachable" >> "$LOG_FILE"
        echo "Port $port: Unreachable"
    fi
done

rm -f /tmp/health_check.json
