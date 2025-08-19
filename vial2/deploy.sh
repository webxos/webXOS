#!/bin/bash

# Deploy script for Vial2 MCP

echo "Starting deployment at $(date)"

# Build Docker image
docker build -t vial2-mcp:latest .

# Login to Docker Hub
echo $DOCKER_PASSWORD | docker login -u $DOCKER_USERNAME --password-stdin

# Push image
docker push vial2-mcp:latest

# Deploy to Netlify
netlify deploy --prod --dir vial2/netlify

echo "Deployment completed at $(date)"

# xAI Artifact Tags: #vial2 #mcp #deploy #script #neon_mcp
