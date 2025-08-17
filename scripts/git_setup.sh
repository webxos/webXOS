#!/bin/bash

set -e

# Load environment variables
source .env

# Check required environment variables
for var in GITHUB_TOKEN NEON_PROJECT_ID; do
  if [ -z "${!var}" ]; then
    echo "Error: $var not set [git_setup.sh:10] [ID:env_error]"
    exit 1
  fi
done

# Create temporary repos directory
mkdir -p /tmp/repos

# Configure git credentials
git config --global user.email "vial-mcp@webxos.netlify.app"
git config --global user.name "Vial MCP"
git config --global credential.helper store
echo "https://${GITHUB_TOKEN}:x-oauth-basic@github.com" > ~/.git-credentials

# Verify git configuration
if git config --global --list | grep -q "user.email"; then
  echo "Git configuration set up successfully"
else
  echo "Error: Git configuration failed [git_setup.sh:25] [ID:git_config_error]"
  exit 1
fi
