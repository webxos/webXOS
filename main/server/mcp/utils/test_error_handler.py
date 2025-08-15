# main/server/mcp/utils/test_error_handler.py
import unittest
from unittest.mock import patch, mock_open
from .error_handler import ErrorHandler
import logging

class TestErrorHandler(unittest.TestCase):
    def setUp(self):
        self.error_handler = ErrorHandler()

    @patch('logging.Logger.error')
    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.record_error')
    def test_handle_generic_error(self, mock_record_error, mock_log_error):
        error = Exception("Test error")
        self.error_handler.handle_generic_error(error, context="test_context")
        mock_log_error.assert_called_with("Error in test_context: Test error")
        mock_record_error.assert_called_with("test_context", "Test error")

    @patch('logging.Logger.error')
    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.record_error')
    def test_handle_wallet_error(self, mock_record_error, mock_log_error):
        error = Exception("Wallet error")
        self.error_handler.handle_wallet_error(error)
        mock_log_error.assert_called_with("Wallet error: Wallet error")
        mock_record_error.assert_called_with("wallet", "Wallet error")

    @patch('logging.Logger.error')
    @patch('main.server.mcp.utils.performance_metrics.PerformanceMetrics.record_error')
    def test_handle_api_error(self, mock_record_error, mock_log_error):
        error = Exception("API error")
        self.error_handler.handle_api_error(error, endpoint="/test")
        mock_log_error.assert_called_with("API error at /test: API error")
        mock_record_error.assert_called_with("api_/test", "API error")

if __name__ == "__main__":
    unittest.main()
