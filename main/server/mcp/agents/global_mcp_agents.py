# main/server/mcp/agents/global_mcp_agents.py
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pydantic import BaseModel
from bson import ObjectId
from ..utils.mcp_error_handler import MCPError
from ..wallet.webxos_wallet import WalletService
import os

class Agent(BaseModel):
    agent_id: str
    vial_id: str
    status: str
    tasks: List[str]
    config: Dict[str, Any]
    user_id: str
    wallet_address: Optional[str] = None

class GlobalMCPAgents:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.collection = self.db["agents"]
        self.wallet_service = WalletService()

    async def create_agent(self, vial_id: str, tasks: List[str], config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        try:
            if not vial_id.startswith("vial"):
                raise MCPError(code=-32602, message="Invalid vial ID: Must start with 'vial'")
            if len(tasks) > 10:
                raise MCPError(code=-32602, message="Maximum 10 tasks allowed")
            wallet_address = await self.wallet_service.create_wallet(user_id)
            agent = {
                "vial_id": vial_id,
                "status": "stopped",
                "tasks": tasks,
                "config": config,
                "user_id": user_id,
                "wallet_address": wallet_address
            }
            result = self.collection.insert_one(agent)
            return {
                "status": "success",
                "agent_id": str(result.inserted_id),
                "wallet_address": wallet_address
            }
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to create agent: {str(e)}")

    async def update_agent_status(self, agent_id: str, status: str, user_id: str) -> Dict[str, Any]:
        try:
            if status not in ["stopped", "running", "training"]:
                raise MCPError(code=-32602, message="Invalid status: Must be stopped, running, or training")
            result = self.collection.update_one(
                {"_id": ObjectId(agent_id), "user_id": user_id},
                {"$set": {"status": status}}
            )
            if result.matched_count == 0:
                raise MCPError(code=-32003, message="Agent not found or access denied")
            return {"status": "success", "agent_id": agent_id}
        except MCPError as e:
            raise e
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to update agent status: {str(e)}")

    async def list_agents(self, user_id: str) -> List[Dict[str, Any]]:
        try:
            agents = self.collection.find({"user_id": user_id}).limit(100)
            return [
                {
                    "agent_id": str(agent["_id"]),
                    "vial_id": agent["vial_id"],
                    "status": agent["status"],
                    "tasks": agent["tasks"],
                    "config": agent["config"],
                    "user_id": agent["user_id"],
                    "wallet_address": agent.get("wallet_address")
                }
                for agent in agents
            ]
        except Exception as e:
            raise MCPError(code=-32603, message=f"Failed to list agents: {str(e)}")

    def close(self):
        self.client.close()
