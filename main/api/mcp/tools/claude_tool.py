from config.config import DatabaseConfig
from lib.code_validator import CodeValidator
from lib.errors import ValidationError, DatabaseError
import logging
from pydantic import BaseModel
from fastapi import HTTPException
from typing import Dict, Any
import importlib.util
import sys
import io
import contextlib

logger = logging.getLogger("mcp.claude")
logger.setLevel(logging.INFO)

class ClaudeExecuteInput(BaseModel):
    code: str
    user_id: str

class ClaudeExecuteOutput(BaseModel):
    result: Any
    output: str
    error: str = None

class ClaudeTool:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.validator = CodeValidator()

    async def execute(self, input: Dict[str, Any]) -> Any:
        try:
            claude_input = ClaudeExecuteInput(**input)
            method = input.get("method", "executeCode")
            
            if method == "executeCode":
                return await self.execute_code(claude_input)
            else:
                raise ValidationError(f"Unknown method: {method}")
        except Exception as e:
            logger.error(f"Claude tool error: {str(e)}")
            raise HTTPException(400, str(e))

    async def execute_code(self, input: ClaudeExecuteInput) -> ClaudeExecuteOutput:
        try:
            # Validate user exists
            user = await self.db.query("SELECT user_id FROM users WHERE user_id = $1", [input.user_id])
            if not user.rows:
                raise ValidationError(f"User not found: {input.user_id}")

            # Validate Claude-generated code
            if not self.validator.is_safe_code(input.code):
                raise ValidationError("Unsafe code detected")

            # Execute code in a safe environment
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                module_name = f"claude_{input.user_id}_{id(input)}"
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                exec(input.code, module.__dict__)

            output = output_buffer.getvalue()
            logger.info(f"Claude code executed for user: {input.user_id}")
            
            # Store execution result in database (optional)
            await self.db.query(
                "INSERT INTO code_executions (user_id, code, output, executed_at) VALUES ($1, $2, $3, $4)",
                [input.user_id, input.code, output, "now()"]
            )

            return ClaudeExecuteOutput(result=None, output=output)
        except Exception as e:
            logger.error(f"Code execution error: {str(e)}")
            return ClaudeExecuteOutput(result=None, output="", error=str(e))
