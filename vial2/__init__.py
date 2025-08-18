from .vial_manager import router as vial_manager_router
from .vial_status import router as vial_status_router
from .vial_config import vial_config
from .vial_migrations import run_migrations
from .vial_proof_of_work import router as pow_router
from .vial_sync import router as sync_router
from .vial_wallet_bridge import router as wallet_bridge_router
from .vial_metrics import router as metrics_router
from .vial_auth import router as auth_router
from .vial_task_queue import router as task_queue_router
from .vial_export import router as export_router
from .vial2_pytorch_controller import router as pytorch_router
from .vial2_offline import router as offline_router
from .vial2_git_ops import router as git_ops_router
from .vial2_security import router as security_router
from .vial2_rate_limit import router as rate_limit_router
from .mcp.inspector.mcp_server import router as mcp_server_router
from .mcp.inspector.tool_manager import router as tool_manager_router
from .mcp.inspector.resource_manager import router as resource_manager_router
from .mcp.inspector.prompt_manager import router as prompt_manager_router
from .mcp.api.mcp_endpoints import router as mcp_endpoints_router
from .mcp.api.git_integration import router as git_integration_router

__all__ = [
    "vial_manager_router",
    "vial_status_router",
    "vial_config",
    "run_migrations",
    "pow_router",
    "sync_router",
    "wallet_bridge_router",
    "metrics_router",
    "auth_router",
    "task_queue_router",
    "export_router",
    "pytorch_router",
    "offline_router",
    "git_ops_router",
    "security_router",
    "rate_limit_router",
    "mcp_server_router",
    "tool_manager_router",
    "resource_manager_router",
    "prompt_manager_router",
    "mcp_endpoints_router",
    "git_integration_router"
]

# xAI Artifact Tags: #vial2 #init #mcp #inspector #pytorch #sqlite #octokit #neon_mcp
