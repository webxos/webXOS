import pytest
from vial2.mcp.langchain import training_orchestrator

def test_orchestrate_training():
    result = training_orchestrator.orchestrate({"task": "train"})
    assert result["status"] == "running"