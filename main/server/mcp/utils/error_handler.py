# main/server/mcp/utils/error_handler.py
from fastapi import HTTPException
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import logging
from typing import Any

logger = logging.getLogger("vial_mcp")
tracer = trace.getTracer("vial_mcp_error_handler")

def handle_auth_error(error: Exception) -> None:
    with tracer.start_as_current_span("handle_auth_error") as span:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))
        logger.error(f"Authentication error: {str(error)}", exc_info=True)
        span.set_attribute("error_type", type(error).__name__)

def handle_wallet_error(error: Exception) -> None:
    with tracer.start_as_current_span("handle_wallet_error") as span:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))
        logger.error(f"Wallet error: {str(error)}", exc_info=True)
        span.set_attribute("error_type", type(error).__name__)

def handle_api_error(error: Exception, endpoint: str) -> None:
    with tracer.start_as_current_span("handle_api_error") as span:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))
        logger.error(f"API error at {endpoint}: {str(error)}", exc_info=True)
        span.set_attribute("error_type", type(error).__name__)
        span.set_attribute("endpoint", endpoint)

def handle_generic_error(error: Exception, context: str = "unknown") -> None:
    with tracer.start_as_current_span("handle_generic_error") as span:
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR))
        logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        span.set_attribute("error_type", type(error).__name__)
        span.set_attribute("context", context)

def raise_http_exception(status_code: int, detail: str, error: Exception = None) -> None:
    with tracer.start_as_current_span("raise_http_exception") as span:
        if error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute("error_type", type(error).__name__)
        logger.error(f"HTTP {status_code}: {detail}")
        raise HTTPException(status_code=status_code, detail=detail)
