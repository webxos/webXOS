import logging
from fastapi import HTTPException, Depends
from typing import Dict, Callable
from datetime import datetime
from ..security_manager import SecurityManager
from ..error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class GatewayRouter:
    """Routes API requests to appropriate services in Vial MCP."""
    def __init__(self, security_manager: SecurityManager, error_handler: ErrorHandler):
        """Initialize GatewayRouter with dependencies.

        Args:
            security_manager (SecurityManager): Manages JWT validation.
            error_handler (ErrorHandler): Handles errors.
        """
        self.security_manager = security_manager
        self.error_handler = error_handler
        self.routes: Dict[str, Callable] = {}
        logger.info("GatewayRouter initialized")

    def register_route(self, endpoint: str, handler: Callable) -> None:
        """Register an API endpoint with its handler.

        Args:
            endpoint (str): API endpoint path.
            handler (Callable): Function to handle the endpoint.

        Raises:
            HTTPException: If endpoint is already registered.
        """
        try:
            if endpoint in self.routes:
                error_msg = f"Endpoint {endpoint} already registered"
                logger.error(error_msg)
                self.error_handler.handle_exception("/api/gateway/register", "system", Exception(error_msg))
            self.routes[endpoint] = handler
            logger.info(f"Registered route {endpoint}")
        except Exception as e:
            self.error_handler.handle_exception("/api/gateway/register", "system", e)

    async def route_request(self, endpoint: str, payload: Dict, access_token: str) -> Dict:
        """Route a request to the appropriate handler.

        Args:
            endpoint (str): API endpoint to route.
            payload (Dict): Request payload.
            access_token (str): JWT access token.

        Returns:
            Dict: Response from the handler.

        Raises:
            HTTPException: If routing fails or endpoint is not found.
        """
        try:
            if endpoint not in self.routes:
                error_msg = f"Endpoint {endpoint} not found"
                logger.error(error_msg)
                self.error_handler.handle_exception(endpoint, payload.get("wallet_id", "anonymous"), Exception(error_msg))
            self.security_manager.validate_token(access_token)
            response = await self.routes[endpoint](payload, access_token)
            logger.info(f"Routed request to {endpoint}")
            return response
        except Exception as e:
            self.error_handler.handle_exception(endpoint, payload.get("wallet_id", "anonymous"), e)
