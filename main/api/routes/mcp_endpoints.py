from fastapi import APIRouter, Depends, HTTPException
from ...security.authentication import verify_token
from ...providers import get_provider
from typing import Dict, Any

router = APIRouter()

@router.get("/api-config")
async def api_config(token: str = Depends(verify_token)):
    return {"rateLimit": 1000, "enabled": True}

@router.post("/void")
async def void_transaction(token: str = Depends(verify_token)):
    return {"message": "Transaction voided"}

@router.post("/mcp/completion")
async def mcp_completion(data: Dict[str, Any], token: str = Depends(verify_token)):
    provider_name = data.get("provider", "xai")
    provider = get_provider(provider_name)
    if not provider:
        raise HTTPException(status_code=400, detail="Invalid provider")
    try:
        result = await provider.completion(data.get("prompt", ""), **data.get("params", {}))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
