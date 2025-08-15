# main/server/mcp/mcp_server.py
from typing import Dict, List, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import asyncio
from ..utils.mcp_error_handler import MCPError, handle_mcp_error
from ..security.security_manager import SecurityManager
from ..notes.mcp_server_notes import NotesService
from ..quantum.mcp_server_quantum import QuantumService
from ..wallet.webxos_wallet import WalletService

app = FastAPI(title="Vial MCP Server")

class Resource(BaseModel):
    uri: str
    name: str
    description: str
    mimeType: str

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: int

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: Any = None
    id: int

class VialMCPServer:
    def __init__(self):
        self.security_manager = SecurityManager()
        self.notes_service = NotesService()
        self.quantum_service = QuantumService()
        self.wallet_service = WalletService()
        self.resources = self.register_resources()
        self.tools = self.register_tools()

    def register_resources(self) -> List[Resource]:
        return [
            Resource(
                uri="vial://notes/{user_id}",
                name="User Notes",
                description="Access to user's note collection",
                mimeType="application/json"
            ),
            Resource(
                uri="vial://quantum/circuits/{vial_id}",
                name="Quantum Circuits",
                description="Quantum circuit definitions and results",
                mimeType="application/vnd.qiskit+json"
            ),
            Resource(
                uri="vial://web3/wallet/{address}",
                name="Web3 Wallet",
                description="Blockchain wallet and transaction data",
                mimeType="application/json"
            )
        ]

    def register_tools(self) -> List[Tool]:
        return [
            Tool(
                name="simulate_quantum_circuit",
                description="Simulate a quantum circuit using Qiskit backend",
                parameters={"circuit_data": "object", "num_shots": "integer"}
            ),
            Tool(
                name="create_note",
                description="Create a new note in the user's collection",
                parameters={"title": "string", "content": "string", "tags": "array"}
            ),
            Tool(
                name="web3_transaction",
                description="Execute a Web3 transaction",
                parameters={"to_address": "string", "amount": "string", "token_contract": "string"}
            )
        ]

    async def get_capabilities(self) -> Dict[str, Any]:
        return {
            "resources": [r.dict() for r in self.resources],
            "tools": [t.dict() for t in self.tools],
            "version": "1.0.0",
            "protocol": "mcp/1.0"
        }

    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        try:
            if request.jsonrpc != "2.0":
                raise MCPError(code=-32600, message="Invalid JSON-RPC version")
            if request.method == "mcp.getCapabilities":
                return MCPResponse(id=request.id, result=await self.get_capabilities())
            elif request.method == "mcp.listResources":
                return MCPResponse(id=request.id, result=[r.dict() for r in self.resources])
            elif request.method == "mcp.callTool":
                tool_name = request.params.get("tool_name")
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if not tool:
                    raise MCPError(code=-32601, message=f"Tool {tool_name} not found")
                if tool_name == "simulate_quantum_circuit":
                    result = await self.quantum_service.simulate_circuit(request.params.get("circuit_data"))
                elif tool_name == "create_note":
                    result = await self.notes_service.create_note(
                        request.params.get("title"), request.params.get("content"), request.params.get("tags")
                    )
                elif tool_name == "web3_transaction":
                    result = await self.wallet_service.send_transaction(
                        request.params.get("to_address"), request.params.get("amount"), request.params.get("token_contract")
                    )
                else:
                    raise MCPError(code=-32601, message="Method not implemented")
                return MCPResponse(id=request.id, result=result)
            else:
                raise MCPError(code=-32601, message="Method not found")
        except MCPError as e:
            return MCPResponse(id=request.id, error={"code": e.code, "message": e.message})

@app.post("/mcp")
async def mcp_endpoint(request: MCPRequest):
    mcp_server = VialMCPServer()
    return await mcp_server.handle_request(request)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
