const { signToken, verifyToken } = require('./auth_manager');
const logger = require('./logger');

async function refreshToken(oldToken) {
  try {
    const decoded = await verifyToken(oldToken);
    if (!decoded) {
      throw new Error('Invalid token');
    }

    const newToken = await signToken({ user_id: decoded.user_id });
    logger.info(`Token refreshed for user: ${decoded.user_id}`);
    return newToken;
  } catch (error) {
    logger.error(`Token refresh error: ${error.message}`);
    throw error;
  }
}

module.exports = { refreshToken };
