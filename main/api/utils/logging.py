import logging
import os

def setup_logging():
    logger = logging.getLogger("NeonMCP")
    logger.setLevel(logging.INFO)
    
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    file_handler = logging.FileHandler("logs/error.log")
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    ))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging()
