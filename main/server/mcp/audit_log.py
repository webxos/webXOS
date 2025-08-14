import logging
import os
import json
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class AuditLog:
    """Manages audit logging for Vial MCP API actions."""
    def __init__(self, log_file: str = "/app/audit_log.jsonl"):
        """Initialize AuditLog with log file path.

        Args:
            log_file (str): Path to audit log file.
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        logger.info("AuditLog initialized")

    def log_action(self, wallet_id: str, endpoint: str, action: str, details: dict = None) -> None:
        """Log an API action to the audit log.

        Args:
            wallet_id (str): Wallet ID performing the action.
            endpoint (str): API endpoint accessed.
            action (str): Action performed (e.g., 'add_note', 'login').
            details (dict, optional): Additional action details.

        Raises:
            HTTPException: If logging fails.
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "wallet_id": wallet_id,
                "endpoint": endpoint,
                "action": action,
                "details": details or {}
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Audit log entry created: {wallet_id} - {action} on {endpoint}")
        except Exception as e:
            logger.error(f"Audit logging failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [AuditLog] Audit logging failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Audit logging failed: {str(e)}")

    def get_audit_logs(self, wallet_id: str = None, limit: int = 100) -> list:
        """Retrieve audit logs, optionally filtered by wallet ID.

        Args:
            wallet_id (str, optional): Filter logs by wallet ID.
            limit (int): Maximum number of log entries to retrieve.

        Returns:
            list: List of audit log entries.

        Raises:
            HTTPException: If log retrieval fails.
        """
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
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
            raise HTTPException(status_code=500, detail=f"Audit log retrieval failed: {str(e)}")
