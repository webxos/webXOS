from fastapi import Request
from ...utils.logging import log_info
import time

async def logging_middleware(request: Request, call_next):
    """Log request details for audit trails."""
    start_time = time.time()
    user_id = getattr(request.state, "user_id", "anonymous")
    log_info(
        f"Request: {request.method} {request.url.path} from {request.client.host} by {user_id}",
        endpoint=request.url.path,
        user_id=user_id
    )
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    log_info(
        f"Response: {request.method} {request.url.path} status {response.status_code} in {duration:.3f}s",
        endpoint=request.url.path,
        user_id=user_id
    )
    
    return response
