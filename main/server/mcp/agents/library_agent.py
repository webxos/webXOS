# main/server/mcp/agents/library_agent.py
from typing import Dict, Any, List
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
from ..utils.cache_manager import CacheManager
import logging
import os
import aiohttp
import json

logger = logging.getLogger("mcp")

class LibraryAgent:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.resources = self.db["resources"]
        self.cache = CacheManager()
        self.github_api_url = "https://api.github.com"
        self.github_token = os.getenv("GITHUB_TOKEN", "")

    @self.metrics.track_request("list_resources")
    async def list_resources(self, agent_id: str, user_id: str) -> List[Dict[str, Any]]:
        try:
            if not agent_id or not user_id:
                raise MCPError(code=-32602, message="Agent ID and user ID are required")
            
            cached = await self.cache.get_cache(f"resources:{user_id}:{agent_id}")
            if cached:
                logger.info(f"Cache hit for resources: {user_id}:{agent_id}")
                return cached
            
            resources = list(self.resources.find({"user_id": user_id, "agent_id": agent_id}))
            for resource in resources:
                resource["_id"] = str(resource["_id"])
            
            await self.cache.set_cache(f"resources:{user_id}:{agent_id}", resources, ttl=300)
            logger.info(f"Listed {len(resources)} resources for user {user_id}, agent {agent_id}")
            return resources
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to list resources: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to list resources: {str(e)}")

    @self.metrics.track_request("sync_github_resource")
    async def sync_github_resource(self, user_id: str, repo_name: str) -> Dict[str, Any]:
        try:
            if not user_id or not repo_name:
                raise MCPError(code=-32602, message="User ID and repository name are required")
            
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.github_token}"}
                async with session.get(
                    f"{self.github_api_url}/repos/{repo_name}", headers=headers
                ) as response:
                    if response.status != 200:
                        raise MCPError(code=-32603, message=f"GitHub API error: {response.status}")
                    repo_data = await response.json()
            
            resource = {
                "user_id": user_id,
                "agent_id": secrets.token_hex(16),
                "type": "github_repo",
                "uri": repo_data["html_url"],
                "metadata": {
                    "name": repo_data["name"],
                    "description": repo_data.get("description", ""),
                    "last_updated": repo_data["updated_at"]
                },
                "created_at": datetime.utcnow()
            }
            self.resources.insert_one(resource)
            await self.cache.delete_cache(f"resources:{user_id}:*")
            logger.info(f"Synced GitHub resource {repo_name} for user {user_id}")
            return {"resource_id": str(resource["_id"]), "status": "synced"}
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Failed to sync GitHub resource: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to sync GitHub resource: {str(e)}")

    def close(self):
        self.client.close()
        self.cache.close()
