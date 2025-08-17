const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');
const { validateTransaction } = require('../../lib/transaction');

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

    const user = await db.query('SELECT wallet_address FROM users WHERE user_id = $1', [user_id]);
    if (!user.rows.length) {
      throw new Error('User not found');
    }

    const transactions = await db.query(
      'SELECT transaction_id, from_address, to_address, amount, created_at FROM transactions WHERE from_address = $1 OR to_address = $1 ORDER BY created_at DESC',
      [user.rows[0].wallet_address]
    );

    const validatedTransactions = transactions.rows.map(tx => ({
      ...tx,
      isValid: validateTransaction(tx)
    }));

    logger.info(`Transaction history retrieved for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ transactions: validatedTransactions })
    };
  } catch (error) {
    logger.error(`Transactions error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
