from pydantic import BaseModel
from typing import Any, Dict, Optional

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: Optional[int | str] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[int | str] = None

class WalletRequest(BaseModel):
    user_id: str

class WalletResponse(BaseModel):
    balance: float
    wallet_key: str
    address: str
    reputation: int

class QuantumState(BaseModel):
    qubits: list
    entanglement: str
