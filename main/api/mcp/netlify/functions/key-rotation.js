const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');
const crypto = require('crypto');

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization'
};

exports.handler = async (event, context) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers };
  }

  try {
    if (event.httpMethod !== 'POST') {
      throw new Error('Method not allowed');
    }

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

    const new_api_key = `WEBXOS-${crypto.randomUUID()}`;
    const new_api_secret = crypto.randomBytes(16).toString('hex');

    await db.query(
      'UPDATE users SET api_key = $1, api_secret = $2 WHERE user_id = $3',
      [new_api_key, new_api_secret, user_id]
    );

    logger.info(`API key rotated for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: 'API key rotated successfully', api_key: new_api_key, api_secret: new_api_secret })
    };
  } catch (error) {
    logger.error(`Key rotation error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
