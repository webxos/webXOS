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
    render build --source . --build-dir $BUILD_DIR
    render deploy --service-name vial-mcp-gateway --build-dir $BUILD_DIR
    ;;
  *)
    echo "Unsupported platform: $PLATFORM"
    exit 1
    ;;
esac
