const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization'
};

exports.handler = async (event, context) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers };
  }

  try {
    if (event.httpMethod !== 'DELETE') {
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

    const user = await db.query('SELECT user_id FROM users WHERE user_id = $1', [user_id]);
    if (!user.rows.length) {
      throw new Error('User not found');
    }

    await db.query('DELETE FROM vials WHERE user_id = $1', [user_id]);
    await db.query('DELETE FROM transactions WHERE from_address = (SELECT wallet_address FROM users WHERE user_id = $1)', [user_id]);
    await db.query('DELETE FROM users WHERE user_id = $1', [user_id]);

    logger.info(`User deleted: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: 'User account deleted successfully' })
    };
  } catch (error) {
    logger.error(`User deletion error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
