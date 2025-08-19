from langchain.adapters.security import SecurityAdapter
from langchain.tools import Tool
from ..error_logging.error_log import error_logger
import logging
import os

logger = logging.getLogger(__name__)

class SecurityTester:
    def __init__(self):
        self.security_adapter = SecurityAdapter(api_key=os.getenv("AUTH_API_KEY", ""))
        self.test_tool = Tool(name="test_injection", func=self._test_injection, description="Tests for injections")

    def _test_injection(self, input_data: str):
        if self.security_adapter.detect_injection(input_data):
            raise ValueError("Potential SQL/XSS injection detected")
        return True

    def test_injection(self, input_data: str):
        try:
            result = self.test_tool.run({"input": input_data})
            logger.info("Security test passed with LangChain tool")
            return result
        except Exception as e:
            error_logger.log_error("security_test", str(e), str(e.__traceback__), sql_statement=None, sql_error_code=None, params={})
            logger.error(f"Security test failed: {str(e)}")
            raise

security_tester = SecurityTester()

# xAI Artifact Tags: #vial2 #mcp #security #tester #langchain #neon_mcp
