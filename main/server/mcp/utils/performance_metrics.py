# main/server/mcp/utils/performance_metrics.py
from opentelemetry import trace, metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource
import jwt
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, Any

class PerformanceMetrics:
    def __init__(self):
        self.tracer = trace.get_tracer("vial_mcp_metrics")
        self.meter = metrics.get_meter("vial_mcp_metrics")
        self.request_duration = self.meter.create_histogram(
            name="request_duration",
            description="Duration of HTTP requests",
            unit="seconds"
        )
        self.jwt_secret = os.getenv("JWT_SECRET", "secret_key")

    @contextmanager
    def track_span(self, span_name: str, attributes: Dict[str, Any] = None):
        span = self.tracer.start_span(span_name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        start_time = datetime.utcnow().timestamp()
        try:
            yield span
        finally:
            duration = datetime.utcnow().timestamp() - start_time
            self.request_duration.record(duration, attributes=attributes or {})
            span.end()

    def create_access_token(self, data: Dict[str, Any], expires_delta: timedelta = timedelta(minutes=30)) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.jwt_secret, algorithm="HS256")

    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except jwt.PyJWTError as e:
            from ..utils.error_handler import handle_auth_error
            handle_auth_error(e)
            raise

metrics.set_meter_provider(MeterProvider(resource=Resource.create(), metric_readers=[PrometheusMetricReader()]))
