from mcp.git.commands import git_commands
from mcp.error_logging.error_log import error_logger
import logging

logger = logging.getLogger(__name__)

class GitHooks:
    def __init__(self):
        self.git = git_commands

    def pre_commit_hook(self):
        try:
            self.git.execute_command("commit")
            logger.info("Pre-commit hook executed")
            return True
        except Exception as e:
            error_logger.log_error("git_pre_commit", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Pre-commit hook failed: {str(e)}")
            raise

git_hooks = GitHooks()

# xAI Artifact Tags: #vial2 #mcp #git #hooks #neon_mcp
