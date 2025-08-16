#!/bin/bash
# Deploy script for Vial MCP Gateway backend
# Usage: ./deploy.sh [render|railway|heroku] [backend-repo-path] [frontend-repo-path]

set -e

# Default paths
BACKEND_REPO=${2:-"/path/to/vial-mcp-gateway"}
FRONTEND_REPO=${3:-"/path/to/webxos"}
BACKEND_URL="https://vial-mcp-backend.onrender.com" # Update with your deployed URL

# Check for deployment platform
PLATFORM=${1:-"render"}
if [[ ! "$PLATFORM" =~ ^(render|railway|heroku)$ ]]; then
  echo "Error: Platform must be 'render', 'railway', or 'heroku'"
  exit 1
fi

# Check if repositories exist
if [ ! -d "$BACKEND_REPO" ]; then
  echo "Error: Backend repository path $BACKEND_REPO does not exist"
  exit 1
fi
if [ ! -d "$FRONTEND_REPO" ]; then
  echo "Error: Frontend repository path $FRONTEND_REPO does not exist"
  exit 1
fi

# Deploy backend
cd "$BACKEND_REPO"
echo "Deploying backend to $PLATFORM..."

if [ "$PLATFORM" = "render" ]; then
  # Render deployment
  if ! command -v render &> /dev/null; then
    echo "Installing Render CLI..."
    curl -sSL https://render.com/cli/install.sh | bash
  fi
  echo "Pushing to Render..."
  render deploy --env production
  BACKEND_URL=$(render service url) # Assumes Render CLI outputs the URL
elif [ "$PLATFORM" = "railway" ]; then
  # Railway deployment
  if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
  fi
  echo "Pushing to Railway..."
  railway up
  BACKEND_URL=$(railway service url) # Assumes Railway CLI outputs the URL
elif [ "$PLATFORM" = "heroku" ]; then
  # Heroku deployment
  if ! command -v heroku &> /dev/null; then
    echo "Installing Heroku CLI..."
    curl https://cli-assets.heroku.com/install.sh | sh
  fi
  echo "Pushing to Heroku..."
  git push heroku main
  BACKEND_URL=$(heroku apps:info -s | grep web_url | cut -d= -f2)
fi

echo "Backend deployed at: $BACKEND_URL"

# Update netlify.toml with backend URL
cd "$FRONTEND_REPO"
echo "Updating netlify.toml with backend URL: $BACKEND_URL"
sed -i "s|http://localhost:8000|$BACKEND_URL|g" main/netlify.toml

# Deploy frontend
if ! command -v netlify &> /dev/null; then
  echo "Installing Netlify CLI..."
  npm install -g netlify-cli
fi
echo "Deploying frontend to Netlify..."
netlify deploy --prod --dir=dist

# Test backend connectivity
echo "Testing backend health check..."
curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/health" | grep -q 200 && echo "Backend health check: OK" || {
  echo "Backend health check: FAILED"
  echo "Error: JSON parse errors will persist until the backend is reachable. Update main/netlify.toml with the correct BACKEND_URL ($BACKEND_URL) and redeploy."
  exit 1
}

# Verify SQLite logging
echo "Checking SQLite error logs..."
if [ -f "$BACKEND_REPO/main/errors.db" ]; then
  sqlite3 "$BACKEND_REPO/main/errors.db" "SELECT * FROM error_logs LIMIT 5"
else
  echo "No error logs found. Backend may not have logged errors yet."
fi

echo "Deployment complete. Frontend: https://api.webxos.netlify.app | Backend: $BACKEND_URL"
echo "Test endpoints with: curl -H 'Authorization: Bearer <token>' $BACKEND_URL/v1/wallet"
