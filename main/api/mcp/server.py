from typing import Dict, Any
from ..utils.logging import log_error, log_info
from ..config.mcp_config import mcp_config
from .handlers.tools import ToolHandler
from .handlers.resources import ResourceHandler
from .handlers.prompts import PromptHandler
from .handlers.tasks import TaskHandler
from ..utils.rag import SmartRAG

class MCPServer:
    def __init__(self):
        self.tool_handler = ToolHandler()
        self.resource_handler = ResourceHandler()
        self.prompt_handler = PromptHandler()
        self.task_handler = TaskHandler()
        self.rag = SmartRAG()
        self.initialized = False

    async def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            server_info = {
                "serverName": mcp_config.MCP_SERVER_NAME,
                "serverVersion": mcp_config.MCP_SERVER_VERSION,
                "protocolVersion": mcp_config.MCP_PROTOCOL_VERSION,
                "capabilities": ["tools", "resources", "prompts", "tasks", "notifications"]
            }
            self.initialized = True
            log_info("MCP server initialized")
            await self.notify("server/initialized", {"status": "success"})
            return server_info
        except Exception as e:
            log_error(f"Initialization failed: {str(e)}")
            raise

    async def initialized(self, params: Dict[str, Any]) -> None:
        log_info("MCP server notified of client initialization")
        await self.notify("client/initialized", params)

    async def list_tools(self) -> list:
        return await self.tool_handler.list_tools()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        return await self.tool_handler.call_tool(name, arguments)

    async def list_resources(self) -> list:
        return await self.resource_handler.list_resources()

    async def read_resource(self, uri: str) -> dict:
        return await self.resource_handler.read_resource(uri)

    async def list_prompts(self) -> list:
        return await self.prompt_handler.list_prompts()

    async def get_prompt(self, name: str, arguments: list) -> str:
        return await self.prompt_handler.get_prompt(name, arguments)

    async def create_task(self, task_id: str, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        return await self.task_handler.create_task(task_id, task_type, params)

    async def notify(self, method: str, params: Dict[str, Any]) -> None:
        from ..utils.logging import notify_message
        await notify_message({"method": method, "params": params})
