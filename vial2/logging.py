import logging
import sys
from ..error_logging.error_log import error_logger

def configure_logging():
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(sys.stderr)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.info("Logging configured to stderr")
    except Exception as e:
        error_logger.log_error("logging_setup", str(e), str(e.__traceback__))
        print(f"Logging setup failed: {str(e)}", file=sys.stderr)
        raise

# xAI Artifact Tags: #vial2 #logging #neon_mcp
