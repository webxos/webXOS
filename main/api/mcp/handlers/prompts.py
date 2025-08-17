from fastapi import HTTPException
from ...utils.logging import log_error, log_info
from ..mcp_schemas import MCPPrompt

class PromptHandler:
    def __init__(self):
        self.prompts = {
            "welcome_prompt": self.get_welcome_prompt(),
            "task_prompt": self.get_task_prompt()
        }

    def get_welcome_prompt(self) -> MCPPrompt:
        return MCPPrompt(
            name="welcome_prompt",
            description="Welcome message for WebXOS MCP Gateway",
            arguments=[{"type": "string", "name": "user_id"}]
        )

    def get_task_prompt(self) -> MCPPrompt:
        return MCPPrompt(
            name="task_prompt",
            description="Prompt for AI coding assistant tasks",
            arguments=[{"type": "string", "name": "task_description"}]
        )

    async def handle_prompt(self, name: str, arguments: list) -> dict:
        if name == "welcome_prompt":
            return await self.handle_welcome_prompt(arguments)
        elif name == "task_prompt":
            return await self.handle_task_prompt(arguments)
        else:
            log_error(f"Prompt {name} not found")
            raise HTTPException(status_code=404, detail=f"Prompt {name} not found")

    async def handle_welcome_prompt(self, arguments: list) -> dict:
        try:
            user_id = arguments[0].get("user_id") if arguments else "unknown"
            log_info(f"Welcome prompt executed for user {user_id}")
            return {"message": f"Welcome to WebXOS MCP Gateway, {user_id}!"}
        except Exception as e:
            log_error(f"Welcome prompt failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prompt error: {str(e)}")

    async def handle_task_prompt(self, arguments: list) -> dict:
        try:
            task_description = arguments[0].get("task_description") if arguments else "No task specified"
            log_info(f"Task prompt executed: {task_description}")
            return {"message": f"Task received: {task_description}"}
        except Exception as e:
            log_error(f"Task prompt failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prompt error: {str(e)}")
