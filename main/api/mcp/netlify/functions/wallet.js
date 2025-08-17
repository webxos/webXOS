const { NeonDB } = require('../../lib/database');
const logger = require('../../lib/logger');
const { verifyToken } = require('../../lib/auth_manager');

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
    const db = new NeonDB();
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
    if (event.httpMethod === 'GET') {
      const user = await db.query('SELECT balance, wallet_address FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length) {
        throw new Error('User not found');
      }

      logger.info(`Wallet retrieved for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({
          balance: user.rows[0].balance,
          address: user.rows[0].wallet_address
        })
      };
    } else if (event.httpMethod === 'POST') {
      const body = event.body ? JSON.parse(event.body) : {};
      const { amount, to_address } = body;
      if (!amount || !to_address) {
        throw new Error('Amount and to_address required');
      }

      const user = await db.query('SELECT balance FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length || user.rows[0].balance < amount) {
        throw new Error('Insufficient balance');
      }

      await db.query('UPDATE users SET balance = balance - $1 WHERE user_id = $2', [amount, user_id]);
      await db.query('UPDATE users SET balance = balance + $1 WHERE wallet_address = $2', [amount, to_address]);
      await db.query(
        'INSERT INTO transactions (transaction_id, from_address, to_address, amount, created_at) VALUES ($1, $2, $3, $4, $5)',
        [`tx_${Date.now()}`, user.rows[0].wallet_address, to_address, amount, new Date()]
      );

      logger.info(`Transfer of ${amount} $WEBXOS from ${user_id} to ${to_address}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ message: 'Transfer successful', amount, to_address })
      };
    } else {
      throw new Error('Method not allowed');
    }
  } catch (error) {
    logger.error(`Wallet error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
