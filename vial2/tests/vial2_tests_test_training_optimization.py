import pytest
from vial2.mcp.langchain import training_optimizer

def test_optimize_model():
    optimized = training_optimizer.optimize({"learning_rate": 0.01})
    assert optimized["learning_rate"] < 0.01