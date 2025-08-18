FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Node.js for MCP Inspector compatibility
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g @modelcontextprotocol/inspector

COPY . .

ENV PYTHONPATH=/app
ENV NEON_DB_URL=postgresql://user:pass@neon.webxos.netlify.app:5432/vial2
ENV GITHUB_CLIENT_ID=your_client_id
ENV GITHUB_CLIENT_SECRET=your_client_secret
ENV NODE_PATH=/usr/lib/node_modules

EXPOSE 8000 6274 6277

CMD ["sh", "-c", "uvicorn vial2.main:app --host 0.0.0.0 --port 8000 & npx @modelcontextprotocol/inspector node mcp/client/mcp_client.js"]

# xAI Artifact Tags: #vial2 #docker #mcp #inspector #sqlite #neon_mcp
