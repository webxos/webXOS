import logging
from datetime import datetime
from ..config.mcp_config import mcp_config
from ..mcp.mcp_schemas import MCPNotification

logging.basicConfig(
    filename='vial_mcp.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def log_info(message: str, notify: bool = False) -> None:
    logger.info(message)
    if notify:
        await notify_message(message)

async def log_error(message: str, notify: bool = False) -> None:
    logger.error(message)
    if notify:
        await notify_message(f"Error: {message}")

async def notify_message(message: str) -> MCPNotification:
    notification = MCPNotification(
        method="notifications/message",
        params={"message": message, "timestamp": datetime.utcnow().isoformat()}
    )
    logger.info(f"Notification sent: {message}")
    return notification
