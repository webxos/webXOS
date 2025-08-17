import subprocess
import os
from lib.code_validator import CodeValidator
import logging

logger = logging.getLogger("mcp.git_hook")
logger.setLevel(logging.INFO)

class GitHook:
    def __init__(self):
        self.validator = CodeValidator()

    def validate_commit(self, repo_path: str, commit_hash: str) -> bool:
        try:
            # Checkout the commit
            subprocess.run(["git", "-C", repo_path, "checkout", commit_hash], check=True)
            
            # Check for agent.py
            agent_file = os.path.join(repo_path, "agent.py")
            if not os.path.exists(agent_file):
                logger.error(f"Agent file not found in commit {commit_hash}")
                return False
            
            # Read and validate code
            with open(agent_file, "r") as f:
                code = f.read()
            
            if not self.validator.is_valid_python(code):
                logger.error(f"Invalid Python code in commit {commit_hash}")
                return False
            
            logger.info(f"Validated commit {commit_hash} successfully")
            return True
        except Exception as e:
            logger.error(f"Error validating commit {commit_hash}: {str(e)}")
            return False
