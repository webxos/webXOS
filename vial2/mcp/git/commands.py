import git
import os
from mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class GitCommands:
    def __init__(self):
        self.repo = git.Repo(os.getenv("GIT_REPO_PATH", "."))

    def execute_command(self, command: str):
        try:
            if command == "commit":
                self.repo.index.add(["*"])
                self.repo.index.commit("Auto-commit from GitCommands")
            logger.info(f"Executed Git command: {command}")
            return True
        except Exception as e:
            error_logger.log_error("git_command", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Git command failed: {str(e)}")
            raise

git_commands = GitCommands()

# xAI Artifact Tags: #vial2 #mcp #git #commands #neon_mcp
