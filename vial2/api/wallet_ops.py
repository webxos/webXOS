from fastapi import APIRouter, HTTPException, Depends
from ..database import get_db
from ..error_logging.error_log import error_logger
from ..security.octokit_oauth import get_octokit_auth
import logging
import hashlib

router = APIRouter(prefix="/mcp/api", tags=["wallet"])

logger = logging.getLogger(__name__)

@router.post("/wallet_ops")
async def wallet_operation(operation: dict, token: str = Depends(get_octokit_auth)):
    try:
        db = await get_db()
        op_type = operation.get("type")
        address = operation.get("address")
        if not op_type or not address:
            raise ValueError("Operation type or address missing")
        
        if op_type == "validate":
            query = "SELECT * FROM wallets WHERE address=$1"
            result = await db.execute(query, address)
            if not result:
                raise ValueError("Wallet not found")
            return {"status": "success", "wallet": result}
        elif op_type == "update_balance":
            amount = operation.get("amount", 0.0)
            query = "UPDATE wallets SET balance = balance + $1 WHERE address = $2 RETURNING balance"
            result = await db.execute(query, amount, address)
            return {"status": "success", "new_balance": result[0]["balance"]}
        else:
            raise ValueError("Unsupported wallet operation")
    except ValueError as e:
        error_logger.log_error("wallet_ops_validation", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Wallet operation validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32602, "message": str(e), "data": operation}
        })
    except Exception as e:
        error_logger.log_error("wallet_ops", str(e), str(e.__traceback__), sql_statement=query, sql_error_code=None, params=operation)
        logger.error(f"Wallet operation failed: {str(e)}")
        raise HTTPException(status_code=400, detail={
            "jsonrpc": "2.0", "error": {"code": -32603, "message": str(e), "data": str(e.__traceback__)}
        })

# xAI Artifact Tags: #vial2 #api #wallet #octokit #sqlite #neon_mcp
