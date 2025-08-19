import pytest
from vial2.mcp.langchain import training_monitor

def test_monitor_progress():
    progress = training_monitor.check_progress({"epoch": 5})
    assert progress["epoch"] == 5