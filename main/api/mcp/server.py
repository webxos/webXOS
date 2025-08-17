import asyncio
import json
from typing import Any, Dict, List
from fastapi import FastAPI
from .mcp_schemas import *
from ..handlers.tools import ToolHandler
from ..handlers.resources import ResourceHandler
from ..handlers.prompts import PromptHandler
from ..handlers.tasks import TaskHandler
from ...utils.logging import log_error, log_info

class MCPServer:
    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self.tasks = {}
        self.capabilities = MCPCapabilities(
            tools={"listChanged": True, "sdk": True},
            resources={"subscribe": True, "listChanged": True},
            prompts={"listChanged": True},
            experimental={"taskHierarchies": True, "walletSync": True, "notifications": True}
        )
        self.tool_handler = ToolHandler()
        self.resource_handler = ResourceHandler()
        self.prompt_handler = PromptHandler()
        self.task_handler = TaskHandler()

    async def initialize(self, params: Dict[str, Any]) -> MCPInitializeResult:
        await self.register_tool("get_wallet_info", self.tool_handler.get_wallet_tool())
        await self.register_resource("quantum://state", self.resource_handler.get_quantum_resource())
        await self.register_resource("decentralized://network", self.resource_handler.get_network_resource())
        await self.register_prompt("welcome_prompt", self.prompt_handler.get_welcome_prompt())
        await self.register_prompt("task_prompt", self.prompt_handler.get_task_prompt())
        log_info("MCP Server initialized")
        return MCPInitializeResult(
            protocolVersion="2024-11-05",
            capabilities=self.capabilities,
            serverInfo=MCPServerInfo(name="webxos-mcp-gateway", version="2.7.8")
        )

    async def initialized(self, params: Dict[str, Any]) -> None:
        log_info("MCP Server initialized handshake completed")

    async def list_tools(self) -> List[MCPTool]:
        return list(self.tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tools:
            log_error(f"Tool {name} not found")
            raise ValueError(f"Tool {name} not found")
        return await self.tool_handler.handle_tool(name, arguments)

    async def list_resources(self) -> List[MCPResource]:
        return list(self.resources.values())

    async def read_resource(self, uri: str) -> Any:
        if uri not in self.resources:
            log_error(f"Resource {uri} not found")
            raise ValueError(f"Resource {uri} not found")
        return await self.resource_handler.handle_resource(uri)

    async def list_prompts(self) -> List[MCPPrompt]:
        return list(self.prompts.values())

    async def get_prompt(self, name: str, arguments: List[Dict[str, Any]]) -> Any:
        if name not in self.prompts:
            log_error(f"Prompt {name} not found")
            raise ValueError(f"Prompt {name} not found")
        return await self.prompt_handler.handle_prompt(name, arguments)

    async def list_tasks(self) -> List[MCPTask]:
        return list(self.tasks.values())

    async def register_tool(self, name: str, tool: MCPTool):
        self.tools[name] = tool
        log_info(f"Tool registered: {name}")

    async def register_resource(self, uri: str, resource: MCPResource):
        self.resources[uri] = resource
        log_info(f"Resource registered: {uri}")

    async def register_prompt(self, name: str, prompt: MCPPrompt):
        self.prompts[name] = prompt
        log_info(f"Prompt registered: {name}")

    async def register_task(self, task_id: str, task: MCPTask):
        self.tasks[task_id] = task
        log_info(f"Task registered: {task_id}")

    async def notify_progress(self, params: Dict[str, Any]):
        log_info(f"Progress notification: {json.dumps(params)}")
        return MCPNotification(method="notifications/progress", params=params)

    async def notify_message(self, message: str):
        log_info(f"Message notification: {message}")
        return MCPNotification(method="notifications/message", params={"message": message})
