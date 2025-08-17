from fastapi import APIRouter, Depends, HTTPException
from ...utils.logging import log_error, log_info
from ...utils.authentication import verify_token
from ...config.redis_config import get_redis
from ...mcp.handlers.tools import ToolHandler
from ...mcp.mcp_schemas import WalletRequest, WalletResponse

router = APIRouter()
tool_handler = ToolHandler()

@router.get("/wallet")
async def get_wallet(user_id: str = Depends(verify_token), redis=Depends(get_redis)):
    try:
        wallet = await tool_handler.handle_wallet_request(user_id, redis)
        log_info(f"Wallet retrieved for user {user_id}")
        return WalletResponse(**wallet)
    except Exception as e:
        log_error(f"Wallet retrieval failed for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Wallet error: {str(e)}")
