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

    const user = await db.query('SELECT wallet_address FROM users WHERE user_id = $1', [user_id]);
    if (!user.rows.length) {
      throw new Error('User not found');
    }

    const blocks = await db.query('SELECT block_id, previous_hash, hash, data, created_at FROM blockchain ORDER BY created_at');
    const localChain = blockchain.getChain();
    const syncedBlocks = [];

    for (const block of blocks.rows) {
      if (!localChain.some(b => b.hash === block.hash)) {
        blockchain.addBlock(block);
        syncedBlocks.push(block);
      }
    }

    await db.query('UPDATE users SET last_sync = $1 WHERE user_id = $2', [new Date(), user_id]);

    logger.info(`Blockchain synced for user: ${user_id}, added ${syncedBlocks.length} blocks`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: 'Blockchain synchronized', syncedBlocks })
    };
  } catch (error) {
    logger.error(`Blockchain sync error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
