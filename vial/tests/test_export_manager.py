import pytest
from vial.export_manager import ExportManager
from unittest.mock import patch
import os

@pytest.fixture
def export_manager():
    return ExportManager()

def test_export_to_markdown(export_manager, tmp_path):
    data = {"results": [{"id": "doc1", "data": "Test data"}]}
    output_path = tmp_path / "output.md"
    export_manager.export_to_markdown(data, str(output_path))
    with open(output_path) as f:
        content = f.read()
    assert "# Exported Data" in content
    assert "Test data" in content

def test_export_to_markdown_invalid_data(export_manager, tmp_path):
    with pytest.raises(ValueError) as exc:
        export_manager.export_to_markdown(None, str(tmp_path / "output.md"))
    assert "Invalid data" in str(exc.value)

def test_export_logging(export_manager, tmp_path):
    error_log = tmp_path / "errorlog.md"
    with open(error_log, "a") as f:
        f.write("")
    data = {"results": [{"id": "doc1", "data": "Test data"}]}
    export_manager.export_to_markdown(data, str(tmp_path / "output.md"))
    with open(error_log) as f:
        log_content = f.read()
    assert "Markdown export by user123" in log_content

def test_export_to_markdown_file_error(export_manager, tmp_path):
    with pytest.raises(OSError) as exc:
        export_manager.export_to_markdown({"results": []}, "/invalid/path/output.md")
    assert "Failed to write file" in str(exc.value)
