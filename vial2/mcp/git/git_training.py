from mcp.git.commands import git_commands
from mcp.langchain.training_orchestrator import training_orchestrator
from mcp.error_logging.error_log import error_logger
import logging
import asyncio

logger = logging.getLogger(__name__)

async def execute_git_training(vial_id: str, commands: list):
    try:
        for cmd in commands:
            git_commands.execute_command(cmd)
            await training_orchestrator.orchestrate_training(vial_id, [cmd])
        logger.info(f"Executed real-time Git training for {vial_id}")
    except Exception as e:
        error_logger.log_error("git_training", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={vial_id})
        logger.error(f"Git training failed: {str(e)}")
        raise

# xAI Artifact Tags: #vial2 #mcp #git #training #neon_mcp
