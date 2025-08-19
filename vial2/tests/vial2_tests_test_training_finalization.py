import pytest
from vial2.mcp.langchain import training_finalizer

def test_finalize_training():
    result = training_finalizer.finalize({"model": "trained"})
    assert result == {"status": "completed"}