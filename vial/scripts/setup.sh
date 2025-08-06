#!/bin/bash
# Setup script for Vial MCP Controller
# Installs dependencies and sets up environment
# Rebuild: Run `chmod +x setup.sh` and `./setup.sh`

echo "Setting up Vial MCP Controller..."

# Create static directory if missing
mkdir -p static uploads/templates uploads/outputs

# Download CDN dependencies
curl -o static/redaxios.min.js https://cdn.jsdelivr.net/npm/redaxios@0.5.1/dist/redaxios.min.js
curl -o static/lz-string.min.js https://cdn.jsdelivr.net/npm/lz-string@1.5.0/libs/lz-string.min.js
curl -o static/mustache.min.js https://cdn.jsdelivr.net/npm/mustache@4.2.0/mustache.min.js
curl -o static/dexie.min.js https://cdn.jsdelivr.net/npm/dexie@3.2.4/dist/dexie.min.js
curl -o static/jwt-decode.min.js https://cdn.jsdelivr.net/npm/jwt-decode@3.1.2/build/jwt-decode.min.js
curl -o static/sql-wasm.wasm https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.8.0/dist/sql-wasm.wasm

# Install Node.js dependencies
npm install express sqlite3 ws jsonwebtoken ajv lz-string

# Copy .env.example to .env if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Update OAUTH_CLIENT_SECRET in .env."
fi

# Create placeholder icon if missing
if [ ! -f static/icon.png ]; then
  touch static/icon.png
  echo "Created placeholder icon.png. Replace with actual icon."
fi

echo "Setup complete. Run 'node src/server.js' or 'docker-compose up' to start."

# Rebuild Instructions: Place in /vial/scripts/. Run `chmod +x setup.sh` and `./setup.sh`. Ensure /vial/.env.example exists. Check /vial/errorlog.md for issues.
