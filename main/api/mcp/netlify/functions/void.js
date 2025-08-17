const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');

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

    await db.query('UPDATE users SET balance = 0, reputation = 0 WHERE user_id = $1', [user_id]);
    await db.query('UPDATE vials SET status = $1, balance = 0 WHERE user_id = $2', ['Stopped', user_id]);
    await db.query('DELETE FROM transactions WHERE from_address = (SELECT wallet_address FROM users WHERE user_id = $1)', [user_id]);

    logger.info(`System voided for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: 'System voided successfully' })
    };
  } catch (error) {
    logger.error(`Void error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
