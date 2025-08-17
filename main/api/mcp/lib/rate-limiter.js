const logger = require('./logger');

class RateLimiter {
  constructor() {
    this.requests = new Map();
    this.limit = 100; // Requests per minute
    this.windowMs = 60 * 1000; // 1 minute
  }

  checkLimit(userId) {
    const now = Date.now();
    const userRequests = this.requests.get(userId) || { count: 0, resetTime: now + this.windowMs };

    if (now > userRequests.resetTime) {
      userRequests.count = 0;
      userRequests.resetTime = now + this.windowMs;
    }

    userRequests.count += 1;
    this.requests.set(userId, userRequests);

    if (userRequests.count > this.limit) {
      logger.warn(`Rate limit exceeded for user: ${userId}`);
      return false;
    }

    logger.info(`Rate limit check passed for user: ${userId}, count: ${userRequests.count}`);
    return true;
  }

  cleanup() {
    const now = Date.now();
    for (const [userId, data] of this.requests) {
      if (now > data.resetTime) {
        this.requests.delete(userId);
      }
    }
  }
}

module.exports = { RateLimiter };
