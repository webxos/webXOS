FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for MCP client compatibility
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models, logs, and error logs
RUN mkdir -p models logs error_logs

# Expose ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/mcp/api/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "mcp.server:app", "--host", "0.0.0.0", "--port", "8000"]

# xAI Artifact Tags: #vial2 #mcp #docker #container #neon_mcp
