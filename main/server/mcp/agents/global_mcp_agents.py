# main/server/mcp/agents/global_mcp_agents.py
from typing import Dict, Any, List
from pymongo import MongoClient
from ..utils.mcp_error_handler import MCPError
from ..utils.performance_metrics import PerformanceMetrics
from ..agents.library_agent import LibraryAgent
from ..agents.translator_agent import TranslatorAgent
import logging
import os

logger = logging.getLogger("mcp")

class GlobalMCPAgents:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
        self.db = self.client["vial_mcp"]
        self.agents = self.db["agents"]
        self.metrics = PerformanceMetrics()
        self.library_agent = LibraryAgent()
        self.translator_agent = TranslatorAgent()

    @self.metrics.track_request("create_agent")
    async def create_agent(self, vial_id: str, tasks: List[str], config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        try:
            if not vial_id or not tasks or not user_id:
                raise MCPError(code=-32602, message="Vial ID, tasks, and user ID are required")
            agent_id = secrets.token_hex(16)
            agent = {
                "agent_id": agent_id,
                "vial_id": vial_id,
                "user_id": user_id,
                "tasks": tasks,
                "config": config,
                "status": "stopped",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            self.agents.insert_one(agent)
            logger.info(f"Created agent {agent_id} for user {user_id}")
            return {"agent_id": agent_id, "status": "created"}
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Agent creation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to create agent: {str(e)}")

    @self.metrics.track_request("execute_workflow")
    async def execute_workflow(self, agent_id: str, workflow_config: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        try:
            agent = self.agents.find_one({"agent_id": agent_id, "user_id": user_id})
            if not agent:
                raise MCPError(code=-32003, message="Agent not found or access denied")
            
            if "manage_resources" in agent["tasks"]:
                resources = await self.library_agent.list_resources(agent_id, user_id)
                workflow_config["resources"] = resources
            
            if "translate" in agent["tasks"]:
                workflow_config = await self.translator_agent.translate_config(workflow_config, "en")
            
            # Simulate workflow execution (e.g., GitHub Actions-like automation)
            workflow_id = f"workflow_{secrets.token_hex(8)}"
            logger.info(f"Executed workflow {workflow_id} for agent {agent_id}")
            return {
                "workflow_id": workflow_id,
                "status": "executed",
                "config": workflow_config
            }
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to execute workflow: {str(e)}")

    @self.metrics.track_request("add_sub_issue")
    async def add_sub_issue(self, parent_issue_id: str, content: str, user_id: str) -> Dict[str, Any]:
        try:
            sub_issue_id = secrets.token_hex(16)
            sub_issue = {
                "sub_issue_id": sub_issue_id,
                "parent_issue_id": parent_issue_id,
                "user_id": user_id,
                "content": content,
                "created_at": datetime.utcnow()
            }
            self.db["sub_issues"].insert_one(sub_issue)
            logger.info(f"Added sub-issue {sub_issue_id} for parent {parent_issue_id}")
            return {"sub_issue_id": sub_issue_id, "status": "created"}
        except MCPError as e:
            raise e
        except Exception as e:
            logger.error(f"Sub-issue creation failed: {str(e)}")
            raise MCPError(code=-32603, message=f"Failed to add sub-issue: {str(e)}")

    def close(self):
        self.client.close()
        self.library_agent.close()
        self.translator_agent.close()
