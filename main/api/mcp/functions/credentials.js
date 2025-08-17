const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');
const crypto = require('crypto');

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
      const user = await db.query('SELECT api_key, api_secret FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length) {
        throw new Error('User not found');
      }

      logger.info(`Credentials retrieved for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
          api_key: user.rows[0].api_key,
          api_secret: user.rows[0].api_secret
        })
      };
    } else if (event.httpMethod === 'POST') {
      const api_key = `WEBXOS-${crypto.randomUUID()}`;
      const api_secret = crypto.randomBytes(16).toString('hex');

      await db.query(
        'UPDATE users SET api_key = $1, api_secret = $2 WHERE user_id = $3',
        [api_key, api_secret, user_id]
      );

      logger.info(`New credentials generated for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
          api_key,
          api_secret
        })
      };
    } else {
      throw new Error('Method not allowed');
    }
  } catch (error) {
    logger.error(`Credentials error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
