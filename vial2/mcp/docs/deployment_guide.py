Deployment Guide
Prerequisites

Docker and Docker Compose installed
Environment variables set (AUTH_API_KEY, GROK_API_KEY, GIT_REPO_PATH, NEON_DB_URL)
Netlify CLI configured for OAuth

Steps

Build Docker Imagedocker build -t vial2-mcp .


Run with Docker Composedocker-compose up --build


Deploy to Netlifynetlify deploy --prod --dir vial2 --functions vial2/netlify/functions


Verify Deploymentcurl -X POST https://your-app.netlify.app/mcp/api/health


Monitor Logs
Check NeonDB for system health and audit logs



xAI Artifact Tags: #vial2 #mcp #docs #deployment #neon_mcp
