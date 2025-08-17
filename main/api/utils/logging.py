import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("webxos_mcp")

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)
