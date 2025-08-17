const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization'
};

exports.handler = async (event, context) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers };
  }

  try {
    const authHeader = event.headers.authorization;
    if (!authHeader) {
      throw new Error('Authorization header missing');
    }

    const token = authHeader.replace('Bearer ', '');
    const decoded = await verifyToken(token);
    if (!decoded) {
      throw new Error('Invalid token');
    }

    const user_id = decoded.user_id;
    const db = new NeonDB();

    if (event.httpMethod === 'GET') {
      const user = await db.query('SELECT user_id, wallet_address, balance, reputation, api_key, api_secret FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length) {
        throw new Error('User not found');
      }

      logger.info(`Profile retrieved for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify(user.rows[0])
      };
    } else if (event.httpMethod === 'POST') {
      const body = event.body ? JSON.parse(event.body) : {};
      const { username, preferences } = body;

      if (!username) {
        throw new Error('Username required');
      }

      await db.query('UPDATE users SET username = $1, preferences = $2 WHERE user_id = $3', [username, JSON.stringify(preferences || {}), user_id]);

      logger.info(`Profile updated for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ message: 'Profile updated successfully', username, preferences })
      };
    } else {
      throw new Error('Method not allowed');
    }
  } catch (error) {
    logger.error(`Profile error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
