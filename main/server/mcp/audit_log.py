import logging
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

class AuditLog:
    """Manages audit logging for Vial MCP system actions."""
    def __init__(self, audit_file: str = "/app/audit_log.jsonl"):
        """Initialize AuditLog with audit file path.

        Args:
            audit_file (str): Path to store audit logs.
        """
        self.audit_file = audit_file
        os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)
        logger.info("AuditLog initialized")

    def log_action(self, wallet_id: str, endpoint: str, action: str, details: dict = None) -> None:
        """Log a system action.

        Args:
            wallet_id (str): Wallet ID performing the action.
            endpoint (str): API endpoint called.
            action (str): Action performed.
            details (dict, optional): Additional action details.
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "wallet_id": wallet_id,
                "endpoint": endpoint,
                "action": action,
                "details": details or {}
            }
            with open(self.audit_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Audit log recorded for {action} by {wallet_id} at {endpoint}")
        except Exception as e:
            logger.error(f"Audit log failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuditLog] Audit log failed: {str(e)}\n")

    def get_audit_logs(self, wallet_id: str = None, limit: int = 100) -> list:
        """Retrieve audit logs, optionally filtered by wallet ID.

        Args:
            wallet_id (str, optional): Filter logs by wallet ID.
            limit (int): Maximum number of log entries to retrieve.

        Returns:
            list: List of audit log entries.
        """
        try:
            logs = []
            if os.path.exists(self.audit_file):
                with open(self.audit_file, "r") as f:
                    lines = f.readlines()[-limit:]
                    for line in lines:
                        log_entry = json.loads(line.strip())
                        if wallet_id is None or log_entry["wallet_id"] == wallet_id:
                            logs.append(log_entry)
            logger.info(f"Retrieved {len(logs)} audit logs for wallet {wallet_id or 'all'}")
            return logs
        except Exception as e:
            logger.error(f"Audit log retrieval failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuditLog] Audit log retrieval failed: {str(e)}\n")
            return []
