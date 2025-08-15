# main/server/mcp/sync/library_sync.py
from typing import Dict, Any, List
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
from ..agents.library_agent import LibraryAgent
import logging
import os
import json

logger = logging.getLogger("mcp")

class LibrarySync:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.sync_log = self.db["sync_log"]
        self.library_agent = LibraryAgent()

    async def sync_resources(self, user_id: str, agent_id: str, external_service: str, external_resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            # Validate input
            if not user_id or not agent_id or not external_service:
                raise MCPError(code=-32602, message="User ID, agent ID, and external service are required")
            
            # Verify agent access
            resources = await self.library_agent.list_resources(agent_id, user_id)
            existing_uris = {r["uri"] for r in resources}
            
            # Sync new resources
            added = 0
            updated = 0
            for resource in external_resources:
                if not all(k in resource for k in ["name", "uri", "type", "metadata"]):
                    continue
                
                if resource["uri"] in existing_uris:
                    # Update existing resource (simplified; real implementation would compare metadata)
                    updated += 1
                else:
                    # Add new resource
                    await self.library_agent.add_resource(
                        agent_id=agent_id,
                        name=resource["name"],
                        uri=resource["uri"],
                        resource_type=resource["type"],
                        metadata=resource["metadata"],
                        user_id=user_id
                    )
                    added += 1
            
            # Log sync operation
            sync_record = {
                "user_id": user_id,
                "agent_id": agent_id,
                "external_service": external_service,
                "added": added,
                "updated": updated,
                "timestamp": datetime.utcnow()
            }
            self.sync_log.insert_one(sync_record)
            
            logger.info(f"Synced {added} new and {updated} updated resources for user {user_id} from {external_service}")
            return {
                "status": "success",
                "added": added,
                "updated": updated,
                "sync_id": str(sync_record["_id"])
            }
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Resource sync failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to sync resources: {str(e)}")

    async def get_sync_history(self, user_id: str, agent_id: str) -> List[Dict[str, Any]]:
        try:
            sync_records = self.sync_log.find({"user_id": user_id, "agent_id": agent_id}).limit(100)
            return [
                {
                    "sync_id": str(r["_id"]),
                    "external_service": r["external_service"],
                    "added": r["added"],
                    "updated": r["updated"],
                    "timestamp": r["timestamp"].isoformat()
                }
                for r in sync_records
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve sync history: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to retrieve sync history: {str(e)}")

    def close(self):
        self.client.close()
        self.library_agent.close()
