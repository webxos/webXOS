# main/server/mcp/utils/error_handler.py
from fastapi import HTTPException
from ..utils.performance_metrics import PerformanceMetrics
import logging
from typing import Any

logging.basicConfig(
    filename='vial_mcp_errors.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ErrorHandler:
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.logger = logging.getLogger("vial_mcp")

    def handle_generic_error(self, error: Exception, context: str) -> None:
        with self.metrics.track_span("handle_generic_error", {"context": context}):
            self.logger.error(f"Error in {context}: {str(error)}")
            self.metrics.record_error(context, str(error))

    def handle_wallet_error(self, error: Exception) -> None:
        with self.metrics.track_span("handle_wallet_error"):
            self.logger.error(f"Wallet error: {str(error)}")
            self.metrics.record_error("wallet", str(error))

    def handle_api_error(self, error: Exception, endpoint: str) -> None:
        with self.metrics.track_span("handle_api_error", {"endpoint": endpoint}):
            self.logger.error(f"API error at {endpoint}: {str(error)}")
            self.metrics.record_error(f"api_{endpoint}", str(error))

def handle_generic_error(error: Exception, context: str) -> None:
    ErrorHandler().handle_generic_error(error, context)

def handle_wallet_error(error: Exception) -> None:
    ErrorHandler().handle_wallet_error(error)

def handle_api_error(error: Exception, endpoint: str) -> None:
    ErrorHandler().handle_api_error(error, endpoint)
