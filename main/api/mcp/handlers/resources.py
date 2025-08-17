from pymongo import MongoClient
from ...config.mcp_config import mcp_config
from ...utils.logging import log_error, log_info

class ResourceHandler:
    def __init__(self):
        self.client = MongoClient(mcp_config.MONGODB_CONNECTION_STRING)
        self.db = self.client[mcp_config.MONGODB_DATABASE]

    async def list_resources(self) -> list:
        try:
            resources = await self.db.resources.find().to_list(100)
            log_info("Resources listed")
            return resources
        except Exception as e:
            log_error(f"Resource listing failed: {str(e)}")
            raise

    async def read_resource(self, uri: str) -> dict:
        try:
            resource = await self.db.resources.find_one({"uri": uri})
            if not resource:
                log_error(f"Resource not found: {uri}")
                raise ValueError(f"Resource not found: {uri}")
            log_info(f"Resource read: {uri}")
            return resource
        except Exception as e:
            log_error(f"Resource read failed: {str(e)}")
            raise

    async def get_quantum_state(self) -> dict:
        try:
            state = await self.db.resources.find_one({"uri": "quantum://state"})
            if not state:
                state = {"qubits": [], "entanglement": "none"}
            log_info("Quantum state retrieved")
            return state
        except Exception as e:
            log_error(f"Quantum state retrieval failed: {str(e)}")
            raise
