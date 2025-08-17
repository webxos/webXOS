const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization'
};

exports.handler = async (event, context) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers };
  }

  try {
    if (event.httpMethod !== 'GET') {
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

    const vials = await db.query('SELECT COUNT(*) as count FROM vials WHERE user_id = $1', [user_id]);
    const transactions = await db.query('SELECT COUNT(*) as count FROM transactions WHERE from_address = (SELECT wallet_address FROM users WHERE user_id = $1)', [user_id]);
    const blocks = await db.query('SELECT COUNT(*) as count FROM blockchain WHERE data->>\'walletAddress\' = (SELECT wallet_address FROM users WHERE user_id = $1)', [user_id]);

    logger.info(`Troubleshoot completed for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        status: 'healthy',
        vials_count: parseInt(vials.rows[0].count),
        transactions_count: parseInt(transactions.rows[0].count),
        blocks_count: parseInt(blocks.rows[0].count)
      })
    };
  } catch (error) {
    logger.error(`Troubleshoot error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
