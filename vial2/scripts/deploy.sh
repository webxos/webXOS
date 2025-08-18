#!/bin/bash

# Build and deploy to Netlify
docker build -t vial2-mcp ./vial2
docker save vial2-mcp | gzip > vial2-mcp.tar.gz
netlify deploy --prod --dir=. --functions=vial2
rm vial2-mcp.tar.gz

# xAI Artifact Tags: #vial2 #deploy #neon_mcp
