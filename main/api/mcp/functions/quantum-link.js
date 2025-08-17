const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
const { Blockchain } = require('../../lib/blockchain');
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
    const blockchain = new Blockchain();
    const body = event.body ? JSON.parse(event.body) : {};
    const { vial_id, work } = body;

    if (!vial_id || !work) {
      throw new Error('Vial ID and work data required');
    }

    const user = await db.query('SELECT wallet_address FROM users WHERE user_id = $1', [user_id]);
    if (!user.rows.length) {
      throw new Error('User not found');
    }

    const vial = await db.query('SELECT id, status FROM vials WHERE id = $1 AND user_id = $2', [vial_id, user_id]);
    if (!vial.rows.length) {
      throw new Error('Vial not found');
    }

    // Simulate quantum link by mining a block
    const block = blockchain.mineBlock(user.rows[0].wallet_address, vial_id, work);
    await db.query(
      'INSERT INTO blockchain (block_id, previous_hash, hash, data, created_at) VALUES ($1, $2, $3, $4, $5)',
      [block.id, block.previousHash, block.hash, JSON.stringify(block.data), new Date()]
    );
    await db.query('UPDATE users SET balance = balance + $1, reputation = reputation + $2 WHERE user_id = $3', [10, 100, user_id]);
    await db.query('UPDATE vials SET status = $1, balance = balance + $2 WHERE id = $3 AND user_id = $4', ['Training', 10, vial_id, user_id]);

    setTimeout(async () => {
      await db.query('UPDATE vials SET status = $1 WHERE id = $2 AND user_id = $3', ['Running', vial_id, user_id]);
      logger.info(`Vial ${vial_id} transitioned to Running for user: ${user_id}`);
    }, 1000);

    logger.info(`Quantum link activated for vial ${vial_id} by user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: 'Quantum link activated', block, reward: 10 })
    };
  } catch (error) {
    logger.error(`Quantum link error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
