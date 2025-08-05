#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Check if the current directory is vial
if [[ ! $(pwd) =~ /vial$ ]]; then
    echo "Please run this script from the vial directory."
    exit 1
fi

# Build Docker image
echo "Building Docker image..."
if ! docker build -t mcp-vial .; then
    echo "Failed to build Docker image."
    exit 1
fi

# Run Docker container
echo "Starting Vial MCP server..."
if ! docker run -d -p 8080:8080 --name mcp-vial-container -e MONGODB_URI=mongodb://host.docker.internal:27017/vial_mcp mcp-vial; then
    echo "Failed to start Docker container."
    exit 1
fi

echo "Vial MCP server running at ws://localhost:8080"
echo "To stop the server, run: docker stop mcp-vial-container"
