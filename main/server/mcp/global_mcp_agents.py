import logging
from fastapi import HTTPException
from .library_agent import LibraryAgent
from .translator_agent import TranslatorAgent

logger = logging.getLogger(__name__)

class GlobalMCPAgents:
    """Manages global agent orchestration for Vial MCP."""
    def __init__(self):
        """Initialize GlobalMCPAgents with supported agents."""
        self.agents = {
            "nomic": LibraryAgent(),  # Placeholder for Nomic agent
            "cognitallmware": LibraryAgent(),  # Placeholder for CogniTALLMware
            "llmware": LibraryAgent(),  # Placeholder for LLMware
            "jina": TranslatorAgent()  # Jina AI for translation
        }
        logger.info("GlobalMCPAgents initialized")

    async def execute_agent_task(self, agent_name: str, task: str, params: dict, access_token: str) -> dict:
        """Execute a task on the specified agent.

        Args:
            agent_name (str): Name of the agent (nomic, cognitallmware, llmware, jina).
            task (str): Task to perform (e.g., 'process_library', 'translate_content').
            params (dict): Task parameters.
            access_token (str): JWT access token.

        Returns:
            dict: Task execution result.

        Raises:
            HTTPException: If task execution fails.
        """
        try:
            if agent_name not in self.agents:
                error_msg = f"Unknown agent: {agent_name}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            agent = self.agents[agent_name]
            if task == "process_library" and agent_name in ["nomic", "cognitallmware", "llmware"]:
                result = await agent.process_library(
                    params.get("library_id"), params.get("wallet_id"),
                    params.get("content"), params.get("db_type"), access_token
                )
            elif task == "translate_content" and agent_name == "jina":
                result = await agent.translate_content(
                    params.get("content"), params.get("target_lang"),
                    params.get("wallet_id"), access_token
                )
            else:
                error_msg = f"Invalid task {task} for agent {agent_name}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail=error_msg)

            logger.info(f"Executed task {task} on agent {agent_name}")
            return result
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Agent task execution failed for {agent_name}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [GlobalMCPAgents] Agent task execution failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Agent task execution failed: {str(e)}")
