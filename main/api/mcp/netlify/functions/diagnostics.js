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

    const user = await db.query('SELECT balance, reputation, wallet_address FROM users WHERE user_id = $1', [user_id]);
    if (!user.rows.length) {
      throw new Error('User not found');
    }

    const vials = await db.query('SELECT id, status, balance FROM vials WHERE user_id = $1', [user_id]);
    const transactions = await db.query('SELECT COUNT(*) as count FROM transactions WHERE from_address = $1 OR to_address = $1', [user.rows[0].wallet_address]);
    const blocks = await db.query('SELECT COUNT(*) as count FROM blockchain');

    const diagnostics = {
      system_status: 'healthy',
      user: {
        balance: user.rows[0].balance,
        reputation: user.rows[0].reputation,
        wallet_address: user.rows[0].wallet_address
      },
      vials: vials.rows,
      transaction_count: parseInt(transactions.rows[0].count),
      block_count: parseInt(blocks.rows[0].count),
      timestamp: new Date().toISOString()
    };

    logger.info(`Diagnostics generated for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify(diagnostics)
    };
  } catch (error) {
    logger.error(`Diagnostics error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
