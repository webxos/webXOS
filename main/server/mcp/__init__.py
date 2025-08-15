# main/server/mcp/__init__.py
"""
Vial MCP Controller Server Package

This package contains the backend services for the Vial MCP Controller,
including authentication, quantum simulation, resource management, and more.
"""

__version__ = "1.0.0"

from .api_gateway import gateway_router, service_registry
from .auth import auth_manager, mcp_server_auth
from .db import db_manager
from .notes import mcp_server_notes
from .quantum import mcp_server_quantum, quantum_simulator
from .resources import mcp_server_resources
from .wallet import webxos_wallet
from .agents import translator_agent, library_agent
from .sync import auth_sync, library_sync
from .utils import (
    error_handler,
    performance_metrics,
    rate_limiter,
    webhook_manager,
    api_config,
    base_prompt,
    health_check,
    api_docs
)
from .events import pubsub_manager

__all__ = [
    "gateway_router",
    "service_registry",
    "auth_manager",
    "mcp_server_auth",
    "db_manager",
    "mcp_server_notes",
    "mcp_server_quantum",
    "quantum_simulator",
    "mcp_server_resources",
    "webxos_wallet",
    "translator_agent",
    "library_agent",
    "auth_sync",
    "library_sync",
    "error_handler",
    "performance_metrics",
    "rate_limiter",
    "webhook_manager",
    "api_config",
    "base_prompt",
    "health_check",
    "api_docs",
    "pubsub_manager"
]
