from fastapi import APIRouter, HTTPException
from ...utils.logging import log_error, log_info
import torch

router = APIRouter()

@router.get("/mcp/rpc")
async def mcp_rpc():
    try:
        # Mock MCP protocol with PyTorch tensor operation
        tensor = torch.rand(3, 3)
        result = {"status": "success", "tensor_output": tensor.tolist()}
        log_info("MCP RPC executed successfully")
        return result
    except Exception as e:
        log_error(f"Traceback: MCP RPC failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"MCP error: {str(e)}")
