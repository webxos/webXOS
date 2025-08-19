import pytest
from vial2.mcp.api import json_logger

def test_log_json():
    log = json_logger.log_json({"event": "test"})
    assert log == {"event": "test", "timestamp": pytest.approx(float)}