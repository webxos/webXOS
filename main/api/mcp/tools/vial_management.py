from config.config import DatabaseConfig
from lib.errors import ValidationError
from lib.code_validator import CodeValidator
from tools.agent_templates import get_all_agents
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any
import hashlib
import subprocess

logger = logging.getLogger("mcp.vial_management")
logger.setLevel(logging.INFO)

class VialExecuteInput(BaseModel):
    user_id: str
    vial_id: str
    code: str

class VialExecuteOutput(BaseModel):
    result: Dict[str, Any]

class VialGitPushInput(BaseModel):
    user_id: str
    vial_id: str
    commit_message: str

class VialGitPushOutput(BaseModel):
    commit_hash: str
    balance: float

class VialManagementTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.validator = CodeValidator()
        self.agents = {agent.get_metadata()["vial_id"]: agent.get_metadata() for agent in get_all_agents()}

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            method = input.get("method", "executeVial")
            if method == "executeVial":
                execute_input = VialExecuteInput(**input)
                return await self.execute_vial(execute_input)
            elif method == "gitPush":
                git_input = VialGitPushInput(**input)
                return await self.git_push(git_input)
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Vial management error: {str(e)}")
            raise HTTPException(400, str(e))

    async def execute_vial(self, input: VialExecuteInput) -> VialExecuteOutput:
        try:
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            if not self.validator.is_valid_python(input.code):
                raise ValidationError("Invalid Python code")
            
            vial = await self.db.query(
                "SELECT vial_id, code FROM vials WHERE user_id = $1 AND vial_id = $2",
                [input.user_id, input.vial_id]
            )
            if not vial.rows:
                raise ValidationError(f"Vial not found: {input.vial_id}")
            
            # Simplified execution (replace with actual PyTorch execution in production)
            result = {"status": "executed", "output": "Code executed successfully"}
            
            await self.db.query(
                "UPDATE vials SET code = $1, webxos_hash = $2 WHERE user_id = $3 AND vial_id = $4",
                [input.code, hashlib.sha256(input.code.encode()).hexdigest(), input.user_id, input.vial_id]
            )
            
            logger.info(f"Executed vial {input.vial_id} for {input.user_id}")
            return VialExecuteOutput(result=result)
        except Exception as e:
            logger.error(f"Execute vial error: {str(e)}")
            raise HTTPException(400, str(e))

    async def git_push(self, input: VialGitPushInput) -> VialGitPushOutput:
        try:
            user = await self.db.query("SELECT user_id, balance FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")
            
            vial = await self.db.query(
                "SELECT vial_id, code FROM vials WHERE user_id = $1 AND vial_id = $2",
                [input.user_id, input.vial_id]
            )
            if not vial.rows:
                raise ValidationError(f"Vial not found: {input.vial_id}")
            
            code = vial.rows[0]["code"]
            if not self.validator.is_valid_python(code):
                raise ValidationError("Invalid Python code in vial")
            
            # Simulate Git push (replace with actual Git repository interaction in production)
            repo_path = f"/tmp/vial_mcp_repo_{input.user_id}_{input.vial_id}"
            try:
                subprocess.run(["git", "init", repo_path], check=True)
                with open(f"{repo_path}/agent.py", "w") as f:
                    f.write(code)
                subprocess.run(["git", "-C", repo_path, "add", "agent.py"], check=True)
                subprocess.run(["git", "-C", repo_path, "commit", "-m", input.commit_message], check=True)
                commit_hash = subprocess.run(
                    ["git", "-C", repo_path, "rev-parse", "HEAD"],
                    capture_output=True, text=True, check=True
                ).stdout.strip()
                
                # Reward for Git push
                current_balance = float(user.rows[0]["balance"])
                reward = 0.5  # Reward for successful push
                new_balance = current_balance + reward
                await self.db.query(
                    "UPDATE users SET balance = $1 WHERE user_id = $2",
                    [new_balance, input.user_id]
                )
                
                logger.info(f"Git push for {input.user_id}, vial {input.vial_id}, commit {commit_hash}")
                return VialGitPushOutput(commit_hash=commit_hash, balance=new_balance)
            finally:
                subprocess.run(["rm", "-rf", repo_path])
        except Exception as e:
            logger.error(f"Git push error: {str(e)}")
            raise HTTPException(400, str(e))
