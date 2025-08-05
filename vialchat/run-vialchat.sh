#!/bin/bash
# Deployment script for VIALCHAT
set -e

# Ensure we're in the vialchat directory
cd "$(dirname "$0")"

# Install dependencies (none needed for static site, but included for completeness)
echo "No external dependencies required for VIALCHAT."

# Validate files
echo "Validating files..."
if [ ! -f "VialChat.html" ] || [ ! -f "static/style.css" ] || [ ! -f "static/neurots.js" ]; then
    echo "Error: Missing required files in /vialchat/"
    exit 1
fi

# Deploy to Netlify (assuming Netlify CLI is installed)
echo "Deploying to Netlify..."
netlify deploy --prod --dir=.

echo "Deployment complete. Access at https://webxos.netlify.app/vialchat/VialChat.html"
