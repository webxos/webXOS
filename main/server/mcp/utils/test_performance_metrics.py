# main/server/mcp/utils/test_performance_metrics.py
import unittest
from unittest.mock import patch
from .performance_metrics import PerformanceMetrics
import jwt
from datetime import datetime, timedelta

class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = PerformanceMetrics()

    @patch('opentelemetry.trace.get_tracer')
    def test_track_span(self, mock_get_tracer):
        mock_span = mock_get_tracer.return_value.start_span.return_value
        with self.metrics.track_span("test_operation", {"key": "value"}):
            pass
        mock_get_tracer.assert_called_with("vial_mcp_tracer")
        mock_span.set_attributes.assert_called_with({"key": "value"})
        mock_span.end.assert_called_once()

    @patch('opentelemetry.metrics.get_meter')
    def test_record_error(self, mock_get_meter):
        mock_counter = mock_get_meter.return_value.create_counter.return_value
        self.metrics.record_error("test_context", "test_error")
        mock_get_meter.assert_called_with("vial_mcp_metrics")
        mock_counter.add.assert_called_with(1, {"context": "test_context", "error": "test_error"})

    def test_verify_token(self):
        token = jwt.encode(
            {"sub": "test_user", "iat": datetime.utcnow(), "exp": datetime.utcnow() + timedelta(minutes=30)},
            "secret_key",
            algorithm="HS256"
        )
        payload = self.metrics.verify_token(token)
        self.assertEqual(payload["sub"], "test_user")

    def test_verify_token_invalid(self):
        with self.assertRaises(jwt.InvalidTokenError):
            self.metrics.verify_token("invalid_token")

if __name__ == "__main__":
    unittest.main()
