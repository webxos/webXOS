import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from config.config import DatabaseConfig
from tools.auth_tool import AuthTool
from tools.wallet import WalletTool
from tools.vial_management import VialManagementTool
from tools.agent_templates import AgentTemplateTool
from lib.security import SecurityHandler
from lib.notifications import NotificationHandler
import logging

logger = logging.getLogger("mcp.server")
logger.setLevel(logging.INFO)

async def main():
    db = DatabaseConfig()
    server = Server("vial-mcp")
    
    # Initialize tools
    auth_tool = AuthTool(db)
    wallet_tool = WalletTool(db)
    vial_management_tool = VialManagementTool(db)
    agent_template_tool = AgentTemplateTool(db)
    security_handler = SecurityHandler(db)
    notification_handler = NotificationHandler(db)
    
    @server.initialize()
    async def initialize():
        logger.info("Initializing Vial MCP server")
        return {
            "serverInfo": {
                "name": "vial-mcp",
                "version": "3.0.0",
                "description": "Vial MCP server for AI agent management"
            }
        }
    
    @server.list_tools()
    async def list_tools():
        return [
            {"name": "authentication", "description": "OAuth authentication and 2FA management"},
            {"name": "wallet", "description": "Wallet management for $WEBXOS"},
            {"name": "blockchain", "description": "Blockchain transaction management"},
            {"name": "health", "description": "Server health monitoring"},
            {"name": "security", "description": "Security event and action logging"},
            {"name": "notifications", "description": "Notification system for critical events"}
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        try:
            if name == "authentication":
                result = await auth_tool.execute(arguments)
            elif name == "wallet":
                result = await wallet_tool.execute(arguments)
            elif name == "blockchain":
                result = await vial_management_tool.execute(arguments)
            elif name == "health":
                result = await server.health()
            elif name == "security":
                result = await security_handler.get_user_actions(arguments)
            elif name == "notifications":
                result = await notification_handler.send_notification(arguments)
            else:
                raise Exception(f"Unknown tool: {name}")
            
            await security_handler.log_user_action(
                user_id=arguments.get("user_id"),
                action=f"tool_call_{name}",
                details={"arguments": arguments}
            )
            logger.info(f"Tool {name} executed for user {arguments.get('user_id')}")
            return result
        except Exception as e:
            logger.error(f"Error executing tool {name}: {str(e)}")
            await security_handler.log_event(
                event_type="tool_execution_error",
                user_id=arguments.get("user_id"),
                details={"tool": name, "error": str(e)}
            )
            raise
    
    async with stdio_server() as streams:
        await server.run(*streams)

if __name__ == "__main__":
    asyncio.run(main())
