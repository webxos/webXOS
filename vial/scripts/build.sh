#!/bin/bash
# Build script for Vial MCP Controller
# Builds Docker image and validates setup
# Rebuild: Run `chmod +x build.sh` and `./build.sh`

echo "Building Vial MCP Controller..."

# Run setup script
./scripts/setup.sh

# Build Docker image
docker build -t vial-mcp-controller .

# Validate static files
for file in redaxios.min.js lz-string.min.js mustache.min.js dexie.min.js jwt-decode.min.js sql-wasm.wasm worker.js icon.png manifest.json; do
  if [ ! -f static/$file ]; then
    echo "[ERROR] Missing static/$file"
    exit 1
  fi
done

echo "Build complete. Run 'docker-compose up' to start."

# Rebuild Instructions: Place in /vial/scripts/. Run `chmod +x build.sh` and `./build.sh`. Ensure /vial/Dockerfile and /vial/static/ files exist. Check /vial/errorlog.md for issues.
