# main/server/mcp/api_gateway/service_registry.py
from typing import Dict, List, Optional
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
import os
import time

class ServiceInstance:
    def __init__(self, service_id: str, address: str, port: int, metadata: Dict[str, str]):
        self.service_id = service_id
        self.address = address
        self.port = port
        self.metadata = metadata
        self.last_heartbeat = time.time()

class ServiceRegistry:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.collection = self.db["services"]
        self.heartbeat_timeout = 30  # Seconds

    async def register_service(self, service_name: str, address: str, port: int, metadata: Dict[str, str]) -> str:
        try:
            if not service_name or not address or not port:
                raise MCPError(code=-32602, message="Service name, address, and port are required")
            service_id = f"{service_name}:{address}:{port}"
            service = {
                "service_id": service_id,
                "service_name": service_name,
                "address": address,
                "port": port,
                "metadata": metadata,
                "last_heartbeat": time.time()
            }
            self.collection.update_one(
                {"service_id": service_id},
                {"$set": service},
                upsert=True
            )
            return service_id
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to register service: {str(e)}")

    async def deregister_service(self, service_id: str) -> None:
        try:
            result = self.collection.delete_one({"service_id": service_id})
            if result.deleted_count == 0:
                raise MCPError(code=-32003, message="Service not found")
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to deregister service: {str(e)}")

    async def update_heartbeat(self, service_id: str) -> None:
        try:
            result = self.collection.update_one(
                {"service_id": service_id},
                {"$set": {"last_heartbeat": time.time()}}
            )
            if result.matched_count == 0:
                raise MCPError(code=-32003, message="Service not found")
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to update heartbeat: {str(e)}")

    async def get_services(self, service_name: str) -> List[Dict[str, any]]:
        try:
            current_time = time.time()
            services = self.collection.find({
                "service_name": service_name,
                "last_heartbeat": {"$gt": current_time - self.heartbeat_timeout}
            })
            return [
                {
                    "service_id": s["service_id"],
                    "address": s["address"],
                    "port": s["port"],
                    "metadata": s["metadata"]
                }
                for s in services
            ]
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to retrieve services: {str(e)}")

    def close(self):
        self.client.close()
