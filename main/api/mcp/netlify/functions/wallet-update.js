const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');
const { validateTransaction } = require('../../lib/transaction');

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
    const body = event.body ? JSON.parse(event.body) : {};
    const { amount, to_address } = body;

    if (!amount || !to_address) {
      throw new Error('Amount and to_address required');
    }

    if (amount <= 0) {
      throw new Error('Amount must be positive');
    }

    const user = await db.query('SELECT wallet_address, balance FROM users WHERE user_id = $1', [user_id]);
    if (!user.rows.length) {
      throw new Error('User not found');
    }

    const from_address = user.rows[0].wallet_address;
    if (user.rows[0].balance < amount) {
      throw new Error('Insufficient balance');
    }

    const transaction = {
      transaction_id: `tx_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`,
      from_address,
      to_address,
      amount,
      created_at: new Date()
    };

    if (!validateTransaction(transaction)) {
      throw new Error('Invalid transaction');
    }

    await db.query('UPDATE users SET balance = balance - $1 WHERE user_id = $2', [amount, user_id]);
    await db.query('INSERT INTO transactions (transaction_id, from_address, to_address, amount, created_at) VALUES ($1, $2, $3, $4, $5)', 
      [transaction.transaction_id, from_address, to_address, amount, transaction.created_at]);

    logger.info(`Wallet updated for user: ${user_id}, transaction: ${transaction.transaction_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: 'Wallet updated successfully', transaction })
    };
  } catch (error) {
    logger.error(`Wallet update error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
