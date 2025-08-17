from pydantic import BaseModel
from typing import Dict, Any, Optional

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: int

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    id: int

class MCPNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPMethods:
    AUTHENTICATION = "authentication"
    VIAL_MANAGEMENT_GET_USER_DATA = "vial-management.getUserData"
    HEALTH_CHECK = "health.check"
    BLOCKCHAIN_GET_INFO = "blockchain.getBlockchainInfo"
    CLAUDE_EXECUTE_CODE = "claude.executeCode"
    WALLET_GET_VIAL_BALANCE = "wallet.getVialBalance"
    WALLET_IMPORT_WALLET = "wallet.importWallet"
    WALLET_EXPORT_VIALS = "wallet.exportVials"
    WALLET_MINE_VIAL = "wallet.mineVial"
