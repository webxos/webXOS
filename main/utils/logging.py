import logging
import os

def setup_logger():
    logger = logging.getLogger("WEBXOS")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("main/vial_mcp.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = setup_logger()

def log_info(message):
    logger.info(message)

def log_error(message):
    logger.error(message)
