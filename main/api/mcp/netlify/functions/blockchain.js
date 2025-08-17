const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const { Blockchain } = require('../../lib/blockchain');
const logger = require('../../lib/logger');

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
    const blockchain = new Blockchain();
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
      const transactions = await db.query('SELECT * FROM transactions WHERE from_address = $1 OR to_address = $1', [user_id]);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ transactions: transactions.rows })
      };
    } else if (event.httpMethod === 'POST') {
      const body = event.body ? JSON.parse(event.body) : {};
      const { vial_id, work } = body;
      if (!vial_id || !work) {
        throw new Error('Vial ID and work required');
      }

      const user = await db.query('SELECT wallet_address FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length) {
        throw new Error('User not found');
      }

      const block = blockchain.mineBlock(user.rows[0].wallet_address, vial_id, work);
      await db.query(
        'INSERT INTO blockchain (block_id, previous_hash, hash, data, created_at) VALUES ($1, $2, $3, $4, $5)',
        [block.id, block.previousHash, block.hash, JSON.stringify(block.data), new Date()]
      );
      await db.query('UPDATE users SET balance = balance + $1 WHERE user_id = $2', [10, user_id]); // PoW reward
      await db.query('UPDATE vials SET balance = balance + $1, status = $2 WHERE id = $3 AND user_id = $4', [10, 'Running', vial_id, user_id]);

      logger.info(`Block mined by ${user_id} for vial ${vial_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ block, reward: 10 })
      };
    } else {
      throw new Error('Method not allowed');
    }
  } catch (error) {
    logger.error(`Blockchain error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
