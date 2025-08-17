from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ..mcp_schemas import MCPNotification
import asyncio
import json

router = APIRouter()

async def sse_generator():
    try:
        while True:
            notification = MCPNotification(
                method="notifications/message",
                params={"message": "Server heartbeat"}
            )
            yield f"data: {json.dumps(notification.dict())}\n\n"
            await asyncio.sleep(30)
    except Exception as e:
        log_error(f"SSE error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@router.get("/notifications")
async def sse_notifications(user_id: str = Depends(verify_token)):
    log_info(f"SSE connection established for user {user_id}")
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
