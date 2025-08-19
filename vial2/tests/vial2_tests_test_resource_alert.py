import pytest
from vial2.mcp.monitoring import resource_alert

def test_check_resources():
    alert = resource_alert.check_resources()
    assert isinstance(alert, dict)
    assert "cpu" in alert