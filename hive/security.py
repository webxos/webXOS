import logging
from logging_config import setup_logging
import re

setup_logging()
logger = logging.getLogger(__name__)

def validate_credentials(api_key: str) -> bool:
    try:
        with open("users.txt", "r") as f:
            for line in f:
                _, _, key = line.strip().split(":")
                if key == api_key:
                    return True
        return False
    except Exception as e:
        logger.error(f"Credential validation failed: {e}")
        return False

def sanitize_input(input_str: str) -> str:
    sanitized = re.sub(r'[<>{}\[\];]', '', input_str)
    sanitized = sanitized.replace('script', '').replace('eval', '')
    return sanitized.strip()