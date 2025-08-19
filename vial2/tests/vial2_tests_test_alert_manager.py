import pytest
from vial2.mcp.monitoring import alert_manager

def test_send_alert():
    alert = alert_manager.send_alert("Test alert")
    assert alert == "Alert sent: Test alert"

def test_get_alerts():
    alerts = alert_manager.get_alerts()
    assert isinstance(alerts, list)
    assert len(alerts) >= 0