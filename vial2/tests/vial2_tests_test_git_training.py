import pytest
from vial2.mcp.langchain import git_training

def test_git_status():
    status = git_training.git_status()
    assert isinstance(status, str)

def test_git_commit():
    result = git_training.git_commit("Test commit")
    assert result == "Commit successful"