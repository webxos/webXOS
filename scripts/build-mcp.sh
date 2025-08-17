#!/bin/bash
set -e

echo "Building Vial MCP Server..."

# Install Python dependencies
pip install -r main/api/mcp/requirements.txt

# Install Node.js dependencies
npm install uuid

# Validate MCP manifest
python -m mcp validate main/mcp.json

# Run tests
pytest main/api/mcp/tests/

# Package for distribution
python setup.py bdist_wheel

echo "Build complete!"
