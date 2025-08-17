from fastapi import HTTPException
from ...utils.logging import log_error, log_info
from ..mcp_schemas import MCPResource, QuantumState

class ResourceHandler:
    def __init__(self):
        self.resources = {
            "quantum://state": self.get_quantum_resource(),
            "decentralized://network": self.get_network_resource()
        }

    def get_quantum_resource(self) -> MCPResource:
        return MCPResource(
            uri="quantum://state",
            name="quantum_state",
            mimeType="application/json",
            description="Current quantum system state"
        )

    def get_network_resource(self) -> MCPResource:
        return MCPResource(
            uri="decentralized://network",
            name="decentralized_network",
            mimeType="application/json",
            description="Decentralized network sync status"
        )

    async def handle_resource(self, uri: str) -> dict:
        if uri == "quantum://state":
            return await self.get_quantum_state()
        elif uri == "decentralized://network":
            return await self.get_network_status()
        else:
            log_error(f"Resource {uri} not found")
            raise HTTPException(status_code=404, detail=f"Resource {uri} not found")

    async def get_quantum_state(self) -> QuantumState:
        try:
            state = QuantumState(
                qubits=[],
                entanglement="initialized"
            )
            log_info("Quantum state retrieved")
            return state
        except Exception as e:
            log_error(f"Quantum state retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Quantum state error: {str(e)}")

    async def get_network_status(self) -> dict:
        try:
            status = {"nodes": 0, "sync_status": "disconnected"}
            log_info("Network status retrieved")
            return status
        except Exception as e:
            log_error(f"Network status retrieval failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
