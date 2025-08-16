import logging
from fastapi import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vial_mcp")

async def logging_middleware(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: {response.status_code}")
    return response
