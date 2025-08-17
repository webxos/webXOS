const { NeonDB } = require('../../lib/database');
const { signToken } = require('../../lib/auth_manager');
const logger = require('../../lib/logger');
const crypto = require('crypto');

const headers = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type'
};

exports.handler = async (event, context) => {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 200, headers };
  }

  try {
    if (event.httpMethod !== 'POST') {
      throw new Error('Method not allowed');
    }

    const body = event.body ? JSON.parse(event.body) : {};
    const { username } = body;

    if (!username) {
      throw new Error('Username required');
    }

    const db = new NeonDB();
    const user_id = `user_${crypto.randomUUID()}`;
    const wallet_address = `wallet_${crypto.randomUUID()}`;
    const api_key = `WEBXOS-${crypto.randomUUID()}`;
    const api_secret = crypto.randomBytes(16).toString('hex');

    const existingUser = await db.query('SELECT user_id FROM users WHERE username = $1', [username]);
    if (existingUser.rows.length) {
      throw new Error('Username already exists');
    }

    await db.query(
      'INSERT INTO users (user_id, username, wallet_address, balance, reputation, api_key, api_secret, created_at) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)',
      [user_id, username, wallet_address, 0, 0, api_key, api_secret, new Date()]
    );

    const token = await signToken({ user_id });
    logger.info(`User registered: ${user_id}, username: ${username}`);
    return {
      statusCode: 201,
      headers,
      body: JSON.stringify({ message: 'User registered successfully', user_id, token, api_key, api_secret })
    };
  } catch (error) {
    logger.error(`Registration error: ${error.message}`);
    return {
      statusCode: error.message.includes('Username') ? 400 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
