from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union

class MCPInitializeRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str = "initialize"
    params: Dict[str, Any]
    id: Optional[Union[int, str]] = None

class MCPInitializedParams(BaseModel):
    pass

class MCPCapabilities(BaseModel):
    tools: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    experimental: Optional[Dict[str, Any]] = None

class MCPServerInfo(BaseModel):
    name: str
    version: str

class MCPInitializeResult(BaseModel):
    protocolVersion: str = "2024-11-05"
    capabilities: MCPCapabilities
    serverInfo: MCPServerInfo

class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

class MCPResource(BaseModel):
    uri: str
    name: str
    mimeType: Optional[str] = None
    description: Optional[str] = None

class MCPPrompt(BaseModel):
    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = []

class MCPTask(BaseModel):
    task_id: str
    name: str
    description: str
    status: str = "pending"
    priority: int = 0
    dependencies: List[str] = []

class MCPNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[Union[int, str]] = None

class WalletRequest(BaseModel):
    user_id: str

class WalletResponse(BaseModel):
    balance: float
    session_balance: float
    wallet_key: str
    address: str
    vial_agent: str
    quantum_state: Dict[str, Any]
    reputation: int
    task_status: str

class QuantumState(BaseModel):
    qubits: List[Any]
    entanglement: str

class OpenAICompatibleTool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]

class OpenAIToolCall(BaseModel):
    id: str
    type: str = "function"
    function: Dict[str, Any]

class ClaudeToolUse(BaseModel):
    name: str
    input: Dict[str, Any]

class ClaudeToolResult(BaseModel):
    content: Union[str, List[Dict[str, Any]]]
    isError: Optional[bool] = False

class GeminiFunctionDeclaration(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
