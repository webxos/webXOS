import git
import os
import logging
from config.config import DatabaseConfig
from lib.security import SecurityHandler

logger = logging.getLogger(__name__)

class GitTool:
    def __init__(self, db: DatabaseConfig, security: SecurityHandler):
        self.db = db
        self.security = security
        self.project_id = "twilight-art-21036984"
        self.repo_base = "/tmp/repos"

    async def execute(self, data: dict) -> dict:
        user_id = data.get("user_id")
        command = data.get("command")
        args = data.get("args", [])
        project_id = data.get("project_id", self.project_id)
        if project_id != self.project_id:
            error_message = f"Invalid Neon project ID: {project_id} [git_tool.py:20] [ID:project_error]"
            logger.error(error_message)
            return {"error": error_message}
        try:
            if command == "clone":
                return await self.clone_repo(user_id, args, project_id)
            elif command == "commit":
                return await self.commit_changes(user_id, args, project_id)
            elif command == "push":
                return await self.push_changes(user_id, args, project_id)
            else:
                error_message = f"Unknown git command: {command} [git_tool.py:25] [ID:git_command_error]"
                logger.error(error_message)
                return {"error": error_message}
        except Exception as e:
            error_message = f"Git command failed: {str(e)} [git_tool.py:30] [ID:git_error]"
            logger.error(error_message)
            await self.security.log_error(user_id, command, error_message)
            return {"error": error_message}

    async def clone_repo(self, user_id: str, args: list, project_id: str) -> dict:
        try:
            repo_url = args[0]
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            repo_path = f"{self.repo_base}/{user_id}/{repo_name}"
            os.makedirs(os.path.dirname(repo_path), exist_ok=True)
            git.Repo.clone_from(repo_url, repo_path)
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "git_clone", json.dumps({"repo_url": repo_url, "repo_name": repo_name}), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Repository cloned for user {user_id}: {repo_url} [git_tool.py:40] [ID:clone_success]")
            return {"status": "success", "repo_path": repo_path}
        except Exception as e:
            error_message = f"Clone failed: {str(e)} [git_tool.py:45] [ID:clone_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def commit_changes(self, user_id: str, args: list, project_id: str) -> dict:
        try:
            repo_path = args[0]
            message = args[1] if len(args) > 1 else "Update from Vial MCP"
            repo = git.Repo(repo_path)
            repo.git.add(A=True)
            repo.index.commit(message)
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "git_commit", json.dumps({"repo_path": repo_path, "message": message}), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Changes committed for user {user_id}: {repo_path} [git_tool.py:55] [ID:commit_success]")
            return {"status": "success", "message": message}
        except Exception as e:
            error_message = f"Commit failed: {str(e)} [git_tool.py:60] [ID:commit_error]"
            logger.error(error_message)
            return {"error": error_message}

    async def push_changes(self, user_id: str, args: list, project_id: str) -> dict:
        try:
            repo_path = args[0]
            branch = args[1] if len(args) > 1 else "main"
            repo = git.Repo(repo_path)
            repo.git.push("origin", branch)
            await self.db.query(
                "INSERT INTO blocks (block_id, user_id, type, data, hash, project_id) VALUES ($1, $2, $3, $4, $5, $6)",
                [str(uuid.uuid4()), user_id, "git_push", json.dumps({"repo_path": repo_path, "branch": branch}), str(uuid.uuid4()), project_id]
            )
            logger.info(f"Changes pushed for user {user_id}: {repo_path} [git_tool.py:70] [ID:push_success]")
            return {"status": "success", "branch": branch}
        except Exception as e:
            error_message = f"Push failed: {str(e)} [git_tool.py:75] [ID:push_error]"
            logger.error(error_message)
            return {"error": error_message}
