const logger = require('./logger');

class Metrics {
  constructor() {
    this.metrics = {
      requests: 0,
      errors: 0,
      latency: [],
      endpoints: new Map()
    };
  }

  recordRequest(endpoint) {
    this.metrics.requests += 1;
    this.metrics.endpoints.set(endpoint, (this.metrics.endpoints.get(endpoint) || 0) + 1);
    logger.info(`Request recorded for endpoint: ${endpoint}, total requests: ${this.metrics.requests}`);
  }

  recordError(endpoint, error) {
    this.metrics.errors += 1;
    logger.error(`Error recorded for endpoint: ${endpoint}, error: ${error.message}`);
  }

  recordLatency(endpoint, duration) {
    this.metrics.latency.push({ endpoint, duration });
    if (this.metrics.latency.length > 1000) {
      this.metrics.latency.shift();
    }
    logger.info(`Latency recorded for endpoint: ${endpoint}, duration: ${duration}ms`);
  }

  getMetrics() {
    const avgLatency = this.metrics.latency.length
      ? this.metrics.latency.reduce((sum, { duration }) => sum + duration, 0) / this.metrics.latency.length
      : 0;
    return {
      total_requests: this.metrics.requests,
      total_errors: this.metrics.errors,
      average_latency: avgLatency.toFixed(2),
      endpoint_counts: Object.fromEntries(this.metrics.endpoints)
    };
  }
}

module.exports = { Metrics };
