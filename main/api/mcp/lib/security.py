import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from config.config import DatabaseConfig
import json
import hashlib
import statistics

logger = logging.getLogger("mcp.security")
logger.setLevel(logging.INFO)

class SecurityHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.anomaly_thresholds = {
            "auth_failure_rate": 5,  # Max failed auth attempts per hour
            "api_request_rate": 50,  # Max API requests per minute
            "cash_out_attempts": 3   # Max cash-out attempts per hour
        }

    async def log_event(self, event_type: str, user_id: Optional[str], details: Dict[str, Any], ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        try:
            await self.db.query(
                """
                INSERT INTO security_events (event_type, user_id, client_id, ip_address, user_agent, details, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    event_type,
                    user_id,
                    details.get("client_id"),
                    ip_address,
                    user_agent,
                    json.dumps(details),
                    datetime.utcnow()
                ]
            )
            logger.info(f"Logged security event: {event_type} for user {user_id}")
            await self.detect_anomalies(event_type, user_id, ip_address)
        except Exception as e:
            logger.error(f"Error logging security event: {str(e)}")

    async def create_session(self, user_id: str) -> str:
        session_id = f"{user_id}:{secrets.token_urlsafe(32)}"
        expires_at = datetime.utcnow() + timedelta(minutes=15)
        try:
            await self.db.query(
                "INSERT INTO sessions (session_key, user_id, expires_at) VALUES ($1, $2, $3)",
                [session_id, user_id, expires_at]
            )
            await self.log_event(
                event_type="session_created",
                user_id=user_id,
                details={"session_id": session_id}
            )
            return session_id
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            await self.log_event(
                event_type="session_creation_error",
                user_id=user_id,
                details={"error": str(e)}
            )
            raise

    async def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await self.db.query(
                "SELECT session_key, user_id, expires_at FROM sessions WHERE session_key = $1",
                [session_id]
            )
            if not session.rows or session.rows[0]["expires_at"] < datetime.utcnow():
                await self.log_event(
                    event_type="session_invalid",
                    user_id=None,
                    details={"session_id": session_id, "reason": "expired or not found"}
                )
                return None
            return {"user_id": session.rows[0]["user_id"], "session_id": session_id}
        except Exception as e:
            logger.error(f"Error validating session: {str(e)}")
            await self.log_event(
                event_type="session_validation_error",
                user_id=None,
                details={"session_id": session_id, "error": str(e)}
            )
            return None

    async def enforce_concurrent_session_limit(self, user_id: str, max_sessions: int = 3):
        try:
            sessions = await self.db.query(
                "SELECT COUNT(*) FROM sessions WHERE user_id = $1 AND expires_at > $2",
                [user_id, datetime.utcnow()]
            )
            if sessions.rows and sessions.rows[0]["count"] >= max_sessions:
                await self.db.query(
                    "DELETE FROM sessions WHERE user_id = $1 AND expires_at = (SELECT MIN(expires_at) FROM sessions WHERE user_id = $2)",
                    [user_id, user_id]
                )
                await self.log_event(
                    event_type="session_limit_enforced",
                    user_id=user_id,
                    details={"max_sessions": max_sessions}
                )
        except Exception as e:
            logger.error(f"Error enforcing session limit: {str(e)}")
            await self.log_event(
                event_type="session_limit_error",
                user_id=user_id,
                details={"error": str(e)}
            )

    async def detect_anomalies(self, event_type: str, user_id: Optional[str], ip_address: Optional[str]):
        try:
            time_window = datetime.utcnow() - timedelta(hours=1)
            if event_type == "auth_error":
                events = await self.db.query(
                    "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND user_id = $2 AND created_at > $3",
                    ["auth_error", user_id, time_window]
                )
                if events.rows[0]["count"] > self.anomaly_thresholds["auth_failure_rate"]:
                    await self.log_event(
                        event_type="anomaly_detected",
                        user_id=user_id,
                        details={"type": "auth_failure_rate", "count": events.rows[0]["count"], "ip_address": ip_address}
                    )
                    logger.warning(f"Anomaly detected: High auth failure rate for user {user_id}")

            elif event_type == "api_request":
                time_window_api = datetime.utcnow() - timedelta(minutes=1)
                events = await self.db.query(
                    "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND user_id = $2 AND created_at > $3",
                    ["api_request", user_id, time_window_api]
                )
                if events.rows[0]["count"] > self.anomaly_thresholds["api_request_rate"]:
                    await self.log_event(
                        event_type="anomaly_detected",
                        user_id=user_id,
                        details={"type": "api_request_rate", "count": events.rows[0]["count"], "ip_address": ip_address}
                    )
                    logger.warning(f"Anomaly detected: High API request rate for user {user_id}")

            elif event_type == "wallet.cashOut":
                events = await self.db.query(
                    "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND user_id = $2 AND created_at > $3",
                    ["wallet.cashOut", user_id, time_window]
                )
                if events.rows[0]["count"] > self.anomaly_thresholds["cash_out_attempts"]:
                    await self.log_event(
                        event_type="anomaly_detected",
                        user_id=user_id,
                        details={"type": "cash_out_attempts", "count": events.rows[0]["count"], "ip_address": ip_address}
                    )
                    logger.warning(f"Anomaly detected: High cash-out attempts for user {user_id}")
        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
