const { NeonDB } = require('../../lib/database');
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
    const db = new NeonDB();
    const user_id = 'a1d57580-d88b-4c90-a0f8-6f2c8511b1e4'; // Mock user for demo
    const user = await db.query('SELECT balance, reputation, wallet_address FROM users WHERE user_id = $1', [user_id]);
    const vials = await db.query('SELECT id, status, balance FROM vials WHERE user_id = $1', [user_id]);

    if (!user.rows.length) {
      throw new Error('User not found');
    }

    logger.info(`Health check for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({
        status: 'healthy',
        balance: user.rows[0].balance,
        reputation: user.rows[0].reputation,
        user_id,
        address: user.rows[0].wallet_address,
        vials: vials.rows
      })
    };
  } catch (error) {
    logger.error(`Health check error: ${error.message}`);
    return {
      statusCode: 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
