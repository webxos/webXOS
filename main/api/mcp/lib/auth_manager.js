const jwt = require('jsonwebtoken');
const logger = require('./logger');

class AuthManager {
  constructor() {
    this.secret = process.env.JWT_SECRET || 'default-secret';
  }

  async verifyApiKey(apiKey) {
    // In production, this would validate against a stored key or external service
    // For demo, accept any non-empty key
    if (!apiKey || apiKey.length < 10) {
      logger.error('Invalid API key');
      return false;
    }
    logger.info(`API key verified: ${apiKey.substring(0, 4)}...`);
    return true;
  }

  async generateToken(apiKey) {
    try {
      const payload = {
        user_id: 'a1d57580-d88b-4c90-a0f8-6f2c8511b1e4', // Mock user ID
        api_key: apiKey,
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 86400 // 24 hours
      };
      const token = jwt.sign(payload, this.secret);
      logger.info('JWT token generated');
      return token;
    } catch (err) {
      logger.error(`Token generation error: ${err.message}`);
      throw err;
    }
  }

  async verifyToken(token) {
    try {
      const decoded = jwt.verify(token, this.secret);
      logger.info(`Token verified for user: ${decoded.user_id}`);
      return decoded;
    } catch (err) {
      logger.error(`Token verification error: ${err.message}`);
      return null;
    }
  }
}

module.exports = { AuthManager, verifyToken: async (token) => new AuthManager().verifyToken(token) };
