from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Any] = None

class MCPInitializeRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str = "initialize"
    params: Dict[str, Any]
    id: Optional[Any] = None

class MCPInitializedParams(BaseModel):
    clientInfo: Dict[str, Any]

class MCPNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]

class ClaudeToolUse(BaseModel):
    id: str
    name: str
    input: Dict[str, Any]

class ClaudeToolResult(BaseModel):
    content: str
    isError: bool = False

class OpenAIToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class GeminiFunctionDeclaration(BaseModel):
    name: str
    parameters: Dict[str, Any]

class WalletRequest(BaseModel):
    user_id: str

class WalletResponse(BaseModel):
    user_id: str
    balance: float
    currency: str
