import logging
import sys
import json
from ..error_logging.error_log import error_logger
from ..config import Config

def configure_logging():
    try:
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%Z"),
                    "level": record.levelname,
                    "module": record.module,
                    "message": record.getMessage(),
                    "node_id": Config.NETLIFY_SITE_ID or "unknown_node"
                }
                return json.dumps(log_entry)

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.StreamHandler(sys.stderr)
            ]
        )
        for handler in logging.getLogger().handlers:
            handler.setFormatter(StructuredFormatter())
        logger = logging.getLogger(__name__)
        logger.info("Structured logging configured to stderr")
    except Exception as e:
        error_logger.log_error("logging_setup", str(e), str(e.__traceback__))
        print(f"Logging setup failed: {str(e)}", file=sys.stderr)
        raise

# xAI Artifact Tags: #vial2 #logging #sqlite #neon_mcp
