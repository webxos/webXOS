from fastapi import HTTPException
from main.api.utils.logging import logger

class MCPErrorHandler:
    @staticmethod
    def handle_validation_error(error: ValueError):
        """Handle validation errors."""
        logger.error(f"Validation error: {str(error)}")
        return HTTPException(status_code=400, detail=str(error))

    @staticmethod
    def handle_authentication_error(error: Exception):
        """Handle authentication errors."""
        logger.error(f"Authentication error: {str(error)}")
        return HTTPException(status_code=401, detail="Authentication failed")

    @staticmethod
    def handle_not_found_error(resource: str):
        """Handle resource not found errors."""
        logger.error(f"Resource not found: {resource}")
        return HTTPException(status_code=404, detail=f"{resource} not found")

    @staticmethod
    def handle_server_error(error: Exception):
        """Handle generic server errors."""
        logger.error(f"Server error: {str(error)}")
        return HTTPException(status_code=500, detail="Internal server error")
