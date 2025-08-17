# main/api/mcp/lib/__init__.py
"""
Vial MCP library module initializer.
This module contains utility libraries for database, authentication, blockchain, and logging.
"""

__version__ = "3.0.0"
__author__ = "Vial MCP Development Team"
__license__ = "MIT"

# Import utility modules for potential Python-based extensions
from . import database
from . import auth_manager
from . import blockchain
from . import logger
