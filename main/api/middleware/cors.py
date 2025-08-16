from fastapi import Request
from fastapi.responses import Response
from ...utils.logging import log_info

ALLOWED_ORIGINS = ["https://api.webxos.netlify.app"]

async def cors_middleware(request: Request, call_next):
    """Custom CORS middleware."""
    origin = request.headers.get("origin")
    response = await call_next(request)
    
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-From"
        log_info(f"CORS allowed for origin: {origin}")
    else:
        log_info(f"CORS denied for origin: {origin}")
    
    return response
