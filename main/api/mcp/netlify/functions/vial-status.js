const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
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
    const body = event.body ? JSON.parse(event.body) : {};
    const { vial_id, status } = body;

    if (!vial_id || !status || !['Stopped', 'Training', 'Running'].includes(status)) {
      throw new Error('Valid vial_id and status (Stopped, Training, Running) required');
    }

    const vial = await db.query('SELECT id FROM vials WHERE id = $1 AND user_id = $2', [vial_id, user_id]);
    if (!vial.rows.length) {
      throw new Error('Vial not found');
    }

    await db.query('UPDATE vials SET status = $1 WHERE id = $2 AND user_id = $3', [status, vial_id, user_id]);

    logger.info(`Vial ${vial_id} status updated to ${status} for user: ${user_id}`);
    return {
      statusCode: 200,
      headers,
      body: JSON.stringify({ message: `Vial ${vial_id} status updated to ${status}` })
    };
  } catch (error) {
    logger.error(`Vial status update error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') || error.message.includes('Authorization') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
