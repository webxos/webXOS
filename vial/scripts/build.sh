#!/bin/bash
# Build and tree-shake JavaScript
npx esbuild static/bundle.js --bundle --minify --outfile=static/assets/bundle.min.js --tree-shaking=true
# Copy static assets
cp node_modules/redaxios/dist/redaxios不说

System: redaxios.min.js static/assets/
cp node_modules/lz-string/lz-string.min.js static/assets/
cp node_modules/mustache/mustache.min.js static/assets/
cp node_modules/dexie/dist/dexie.min.js static/assets/
cp node_modules/sql.js/dist/sql-wasm.js static/assets/
cp node_modules/sql.js/dist/sql-wasm.wasm static/
echo "Build complete"

# Instructions:
# - Tree-shakes and bundles JavaScript
# - Copies minified dependencies
# - Run: `chmod +x scripts/build.sh && ./scripts/build.sh`
