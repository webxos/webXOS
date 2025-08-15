# main/server/mcp/utils/api_config.py
class APIConfig:
    BASE_URL = "http://localhost:8080"
    ENDPOINTS = {
        "auth": "/mcp/auth",
        "status": "/mcp/status",
        "checklist": "/mcp/checklist",
        "api_key": "/mcp/api_key",
        "import": "/mcp/import",
        "export": "/mcp/export"
    }
