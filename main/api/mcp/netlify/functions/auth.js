const { AuthManager } = require('../../lib/auth_manager');
const { NeonDB } = require('../../lib/database');
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
    const authManager = new AuthManager();
    const db = new NeonDB();
    const body = event.body ? JSON.parse(event.body) : {};

    if (event.httpMethod !== 'POST') {
      throw new Error('Method not allowed');
    }

    const { api_key } = body;
    if (!api_key) {
      throw new Error('API key required');
    }

    const isValid = await authManager.verifyApiKey(api_key);
    if (!isValid) {
      throw new Error('Invalid API key');
    }

    const token = await authManager.generateToken(api_key);
    const user = await db.query('SELECT * FROM users WHERE api_key = $1', [api_key]);

    if (!user.rows.length) {
      const user_id = 'a1d57580-d88b-4c90-a0f8-6f2c8511b1e4';
      const wallet_address = 'e8aa2491-f9a4-4541-ab68-fe7a32fb8f1d';
      await db.query(
        'INSERT INTO users (user_id, api_key, api_secret, balance, reputation, wallet_address, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7)',
        [user_id, api_key, 'MOCKSECRET1234567890', 38940.0000, 1200983581, wallet_address, new Date()]
      );
      logger.info(`New user created: ${user_id}`);
    }

    logger.info(`Auth successful for user: vial_user`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ access_token: token, token_type: 'bearer', expires_in: 86400 })
    };
  } catch (error) {
    logger.error(`Auth error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
