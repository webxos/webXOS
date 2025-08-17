#!/bin/bash

# Deploy WEBXOS MCP Gateway to Render
if [ $# -ne 3 ]; then
  echo "Usage: $0 <platform> <source_dir> <build_dir>"
  exit 1
fi

PLATFORM=$1
SOURCE_DIR=$2
BUILD_DIR=$3

case $PLATFORM in
  "render")
    echo "Deploying to Render..."
    cd $SOURCE_DIR || exit 1
    # Install dependencies
    pip install -r requirements.txt
    # Build Docker image
    docker build -t vial-mcp-gateway .
    # Deploy to Render
    render deploy --service-name vial-mcp-gateway --build-dir $BUILD_DIR
    # Verify deployment
    curl -f https://vial-mcp-backend.onrender.com/v1/health || {
      echo "Deployment verification failed"
      exit 1
    }
    echo "Deployment successful"
    ;;
  *)
    echo "Unsupported platform: $PLATFORM"
    exit 1
    ;;
esac
