from fastapi import APIRouter, Depends, HTTPException, WebSocket
from config.config import DatabaseConfig
from lib.security import SecurityHandler
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
import json

logger = logging.getLogger("mcp.monitoring")
logger.setLevel(logging.INFO)

router = APIRouter()

class MonitoringHandler:
    def __init__(self, db: DatabaseConfig):
        self.db = db
        self.security_handler = SecurityHandler(db)

    async def get_security_kpis(self, time_window_hours: int = 24) -> Dict[str, Any]:
        try:
            time_window = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            # Authentication success/failure rates
            auth_success = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["auth_success", time_window]
            )
            auth_failure = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["auth_error", time_window]
            )
            
            # Token validation latency (simplified as event count)
            token_validations = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["token_verified", time_window]
            )
            
            # Session management efficiency
            active_sessions = await self.db.query(
                "SELECT COUNT(*) FROM sessions WHERE expires_at > $1",
                [datetime.utcnow()]
            )
            
            # Security incident response times (simplified as anomaly detections)
            anomalies = await self.db.query(
                "SELECT COUNT(*) FROM security_events WHERE event_type = $1 AND created_at > $2",
                ["anomaly_detected", time_window]
            )
            
            kpis = {
                "auth_success_rate": auth_success.rows[0]["count"] / (auth_success.rows[0]["count"] + auth_failure.rows[0]["count"] + 1) * 100,
                "auth_failure_count": auth_failure.rows[0]["count"],
                "token_validations": token_validations.rows[0]["count"],
                "active_sessions": active_sessions.rows[0]["count"],
                "anomalies_detected": anomalies.rows[0]["count"]
            }
            
            await self.security_handler.log_event(
                event_type="monitoring_kpis",
                user_id=None,
                details={"kpis": kpis, "time_window_hours": time_window_hours}
            )
            logger.info(f"Generated security KPIs for last {time_window_hours} hours")
            return kpis
        except Exception as e:
            logger.error(f"Error generating KPIs: {str(e)}")
            await self.security_handler.log_event(
                event_type="monitoring_error",
                user_id=None,
                details={"error": str(e)}
            )
            raise HTTPException(status_code=500, detail=str(e))

    async def stream_kpis(self, websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                kpis = await self.get_security_kpis(time_window_hours=1)
                await websocket.send_json({"type": "kpi_update", "data": kpis})
                await asyncio.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error streaming KPIs: {str(e)}")
            await self.security_handler.log_event(
                event_type="kpi_stream_error",
                user_id=None,
                details={"error": str(e)}
            )
            await websocket.close()

@router.get("/monitoring/kpis")
async def get_kpis(time_window_hours: int = 24, handler: MonitoringHandler = Depends(lambda: MonitoringHandler(DatabaseConfig()))):
    return await handler.get_security_kpis(time_window_hours)

@router.websocket("/monitoring/kpis/stream")
async def stream_kpis(websocket: WebSocket, handler: MonitoringHandler = Depends(lambda: MonitoringHandler(DatabaseConfig()))):
    await handler.stream_kpis(websocket)
