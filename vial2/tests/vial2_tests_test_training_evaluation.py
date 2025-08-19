import pytest
from vial2.mcp.langchain import training_evaluator

def test_evaluate_model():
    score = training_evaluator.evaluate({"accuracy": 0.95})
    assert score > 0.9