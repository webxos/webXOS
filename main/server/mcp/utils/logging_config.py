import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/vial_mcp.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('VialMCP')
