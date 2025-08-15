# main/server/mcp/scripts/deploy.sh
#!/bin/bash

# Deployment script for Vial MCP Controller
set -e

# Configuration
IMAGE_NAME="vial-mcp-controller"
TAG="latest"
REGISTRY="docker.io"
NAMESPACE="mcp"
KUBECTL="kubectl"

# Build Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${TAG} -f ../Dockerfile .

# Tag and push to registry
echo "Pushing Docker image to registry..."
docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}
docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
${KUBECTL} apply -f ../kubernetes/manifests/mcp_deployment.yaml -n ${NAMESPACE}

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
${KUBECTL} wait --for=condition=available --timeout=300s deployment/vial-mcp-controller -n ${NAMESPACE}

# Verify deployment
echo "Verifying deployment..."
${KUBECTL} get pods -n ${NAMESPACE} -l app=vial-mcp-controller

# Check health endpoint
echo "Checking health endpoint..."
sleep 5  # Wait for service to be available
SERVICE_IP=$(${KUBECTL} get service vial-mcp-service -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
curl -f http://${SERVICE_IP}/health || {
    echo "Health check failed"
    exit 1
}

echo "Deployment completed successfully"
