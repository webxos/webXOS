import logging
from typing import Dict, Any, Optional
from config.config import DatabaseConfig, SMTPConfig
from email.mime.text import MIMEText
import smtplib
import asyncio
from datetime import datetime
from pydantic import BaseModel

logger = logging.getLogger("mcp.notifications")
logger.setLevel(logging.INFO)

class NotificationInput(BaseModel):
    user_id: str
    notification_type: str
    details: Dict[str, Any]

class NotificationHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.smtp_config = SMTPConfig()

    async def send_notification(self, input: NotificationInput):
        try:
            # Store notification in database
            await self.db.query(
                """
                INSERT INTO notifications (user_id, notification_type, details, created_at)
                VALUES ($1, $2, $3, $4)
                """,
                [
                    input.user_id,
                    input.notification_type,
                    json.dumps(input.details),
                    datetime.utcnow()
                ]
            )
            
            # Send email notification for critical events
            if input.notification_type in ["anomaly_detected", "data_erasure"]:
                subject = f"Vial MCP: {input.notification_type.replace('_', ' ').title()}"
                body = f"Notification for user {input.user_id}:\n\nType: {input.notification_type}\nDetails: {json.dumps(input.details, indent=2)}"
                
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = self.smtp_config.smtp_user
                msg["To"] = self.smtp_config.alert_email
                
                with smtplib.SMTP(self.smtp_config.smtp_server, self.smtp_config.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_config.smtp_user, self.smtp_config.smtp_password)
                    server.send_message(msg)
                
                logger.info(f"Sent notification email for {input.notification_type} to user {input.user_id}")
            
            logger.info(f"Stored notification for user {input.user_id}: {input.notification_type}")
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            raise

    async def monitor_audit_logs(self):
        """Monitor audit logs for critical events and send notifications."""
        try:
            while True:
                # Check for recent critical audit log entries
                critical_actions = await self.db.query(
                    """
                    SELECT user_id, action, details FROM audit_logs 
                    WHERE action IN ('anomaly_detected', 'data_erasure') 
                    AND created_at > $1
                    AND notified IS NULL
                    """,
                    [datetime.utcnow() - timedelta(minutes=5)]
                )
                
                for row in critical_actions.rows:
                    await self.send_notification(NotificationInput(
                        user_id=row["user_id"],
                        notification_type=row["action"],
                        details=json.loads(row["details"])
                    ))
                    await self.db.query(
                        "UPDATE audit_logs SET notified = TRUE WHERE user_id = $1 AND action = $2 AND created_at = $3",
                        [row["user_id"], row["action"], row["created_at"]]
                    )
                
                await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error monitoring audit logs: {str(e)}")
