import pytest
from vial2.mcp.security import security_tester

def test_security_scan():
    result = security_tester.run_scan()
    assert result["status"] == "secure"