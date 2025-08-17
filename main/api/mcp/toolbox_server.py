import os
import subprocess
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import yaml
from typing import List, Dict, Any
import psycopg2
from redis import Redis
import opentelemetry.trace as trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces")
trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))

# FastAPI app
app = FastAPI(title="Vial MCP Toolbox Server")
security = HTTPBearer()

# Load tools.yaml
with open(os.path.join(os.path.dirname(__file__), '../../config/tools.yaml'), 'r') as f:
    config = yaml.safe_load(f)

# Database connections
db_connections = {}
redis_client = None

def init_connections():
    global redis_client
    for source_name, source_config in config['sources'].items():
        if source_config['kind'] == 'postgres':
            db_connections[source_name] = psycopg2.connect(
                host=source_config['host'],
                port=source_config['port'],
                database=source_config['database'],
                user=source_config['user'],
                password=source_config['password'],
                sslmode=source_config.get('ssl', 'require')
            )
        elif source_config['kind'] == 'redis':
            redis_client = Redis.from_url(source_config['url'], decode_responses=True)

# Pydantic models
class ToolParameter(BaseModel):
    name: str
    value: str

class ToolExecutionRequest(BaseModel):
    parameters: List[ToolParameter]

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    expected_key = os.getenv('API_KEY')
    if credentials.credentials != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Get toolsets
@app.get("/mcp/toolsets")
async def get_toolsets(token: str = Depends(verify_token)):
    with tracer.start_as_current_span("get_toolsets"):
        return {"tools": list(config['tools'].keys()), "toolsets": config['toolsets']}

# Execute tool
@app.post("/mcp/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, request: ToolExecutionRequest, token: str = Depends(verify_token)):
    with tracer.start_as_current_span(f"execute_tool_{tool_name}"):
        if tool_name not in config['tools']:
            raise HTTPException(status_code=404, detail="Tool not found")
        tool = config['tools'][tool_name]
        source = db_connections.get(tool['source'])
        if not source:
            raise HTTPException(status_code=400, detail="Invalid source")
        try:
            with source.cursor() as cursor:
                params = [param.value for param in request.parameters]
                cursor.execute(tool['statement'], params)
                result = cursor.fetchall()
                source.commit()
                return {"result": result}
        except Exception as e:
            source.rollback()
            logging.error(f"Tool execution failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

# Start server
if __name__ == "__main__":
    init_connections()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
