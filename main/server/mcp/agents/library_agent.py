# main/server/mcp/agents/library_agent.py
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
from ..agents.global_mcp_agents import GlobalMCPAgents
import os
import requests

class LibraryResource(BaseModel):
    resource_id: str
    name: str
    uri: str
    type: str
    metadata: Dict[str, Any]

class LibraryAgent:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.collection = self.db["library_resources"]
        self.global_agents = GlobalMCPAgents()
        self.resource_api_url = os.getenv("RESOURCE_API_URL", "https://api.example.com/resources")

    async def add_resource(self, agent_id: str, name: str, uri: str, resource_type: str, metadata: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        try:
            # Validate agent and user
            agents = await self.global_agents.list_agents(user_id)
            agent = next((a for a in agents if a["agent_id"] == agent_id), None)
            if not agent:
                raise MCPError(code=-32003, message="Agent not found or access denied")
            
            # Validate input
            if not name or not uri or not resource_type:
                raise MCPError(code=-32602, message="Name, URI, and resource type are required")
            if resource_type not in ["dataset", "model", "document"]:
                raise MCPError(code=-32602, message="Unsupported resource type")

            # Verify resource accessibility (mocked for simplicity)
            response = requests.head(uri)
            if response.status_code != 200:
                raise MCPError(code=-32603, message="Resource URI is not accessible")

            resource = {
                "agent_id": agent_id,
                "user_id": user_id,
                "name": name,
                "uri": uri,
                "type": resource_type,
                "metadata": metadata
            }
            result = self.collection.insert_one(resource)
            return {
                "status": "success",
                "resource_id": str(result.inserted_id)
            }
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to add resource: {str(e)}")

    async def list_resources(self, agent_id: str, user_id: str, resource_type: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            agents = await self.global_agents.list_agents(user_id)
            if not any(a["agent_id"] == agent_id for a in agents):
                raise MCPError(code=-32003, message="Agent not found or access denied")
            
            query = {"agent_id": agent_id, "user_id": user_id}
            if resource_type:
                query["type"] = resource_type
            resources = self.collection.find(query).limit(100)
            return [
                {
                    "resource_id": str(r["_id"]),
                    "name": r["name"],
                    "uri": r["uri"],
                    "type": r["type"],
                    "metadata": r["metadata"]
                }
                for r in resources
            ]
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to list resources: {str(e)}")

    def close(self):
        self.client.close()
