import logging
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List
from ..db.db_manager import DatabaseManager
from ..security_manager import SecurityManager
from ..error_handler import ErrorHandler
from .translator_agent import TranslatorAgent
from .library_agent import LibraryAgent

logger = logging.getLogger(__name__)

class AgentTaskRequest(BaseModel):
    wallet_id: str
    task_type: str
    parameters: dict

class GlobalMCPAgents:
    """Orchestrates agent tasks for Vial MCP."""
    def __init__(self, db_manager: DatabaseManager = None, security_manager: SecurityManager = None, error_handler: ErrorHandler = None):
        """Initialize GlobalMCPAgents with dependencies.

        Args:
            db_manager (DatabaseManager): Database manager instance.
            security_manager (SecurityManager): Security manager instance.
            error_handler (ErrorHandler): Error handler instance.
        """
        self.db_manager = db_manager or DatabaseManager()
        self.security_manager = security_manager or SecurityManager()
        self.error_handler = error_handler or ErrorHandler()
        self.agents = {
            "translator": TranslatorAgent(self.db_manager),
            "library": LibraryAgent(self.db_manager)
        }
        logger.info("GlobalMCPAgents initialized")

    async def execute_task(self, request: AgentTaskRequest, access_token: str) -> dict:
        """Execute a task using the specified agent.

        Args:
            request (AgentTaskRequest): Task execution request.
            access_token (str): JWT access token.

        Returns:
            dict: Task execution result.

        Raises:
            HTTPException: If the operation fails.
        """
        try:
            payload = self.security_manager.validate_token(access_token)
            if payload["wallet_id"] != request.wallet_id:
                error_msg = "Unauthorized wallet access"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/agents/execute", request.wallet_id, Exception(error_msg))
            agent = self.agents.get(request.task_type)
            if not agent:
                error_msg = f"Unknown agent type: {request.task_type}"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/agents/execute", request.wallet_id, Exception(error_msg))
            result = await agent.process_task(request.parameters)
            task_id = await self.db_manager.log_task(request.wallet_id, request.task_type, result)
            logger.info(f"Executed task {task_id} for wallet {request.wallet_id} with agent {request.task_type}")
            return {"task_id": task_id, "result": result}
        except Exception as e:
            self.error_handler.handle_exception("/api/agents/execute", request.wallet_id, e)
