const { NeonDB } = require('../../lib/database');
const { verifyToken } = require('../../lib/auth_manager');
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

    if (event.httpMethod === 'GET') {
      const user = await db.query('SELECT balance, wallet_address, reputation FROM users WHERE user_id = $1', [user_id]);
      if (!user.rows.length) {
        throw new Error('User not found');
      }

      const vials = await db.query('SELECT id, status, balance, wallet_address FROM vials WHERE user_id = $1', [user_id]);
      const data = `# Vial MCP Export\n\n## Wallet\n- Balance: ${user.rows[0].balance.toFixed(4)} $WEBXOS\n- Address: ${user.rows[0].wallet_address}\n- User ID: ${user_id}\n- Reputation: ${user.rows[0].reputation}\n\n## Vials\n${vials.rows.map(vial => `# Vial ${vial.id}\n- Status: ${vial.status}\n- Balance: ${vial.balance.toFixed(4)} $WEBXOS\n- Wallet Address: ${vial.wallet_address || 'none'}\n`).join('---\n')}`;

      logger.info(`Export data generated for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ export_data: data })
      };
    } else if (event.httpMethod === 'POST') {
      const body = event.body ? JSON.parse(event.body) : {};
      const { import_data } = body;
      if (!import_data) {
        throw new Error('Import data required');
      }

      const lines = import_data.split('\n');
      let currentVial = null;
      const vials = [];
      let wallet = { balance: 0, address: null, reputation: 0 };

      for (let line of lines) {
        if (line.match(/^- Balance: ([\d.]+) \$WEBXOS/)) {
          wallet.balance = parseFloat(line.match(/^- Balance: ([\d.]+) \$/)[1]) || 0;
        } else if (line.match(/^- Address: ([\w-]+)/)) {
          wallet.address = line.match(/^- Address: ([\w-]+)/)[1];
        } else if (line.match(/^- Reputation: (\d+)/)) {
          wallet.reputation = parseInt(line.match(/^- Reputation: (\d+)/)[1]) || 0;
        } else if (line.match(/^# Vial (vial\d)/)) {
          if (currentVial) vials.push(currentVial);
          currentVial = { id: line.match(/^# Vial (vial\d)/)[1], status: 'Stopped', balance: 0, wallet_address: null };
        } else if (line.match(/^- Status: (\w+)/)) {
          currentVial.status = line.match(/^- Status: (\w+)/)[1];
        } else if (line.match(/^- Balance: ([\d.]+) \$WEBXOS/)) {
          currentVial.balance = parseFloat(line.match(/^- Balance: ([\d.]+) \$/)[1]) || 0;
        } else if (line.match(/^- Wallet Address: ([\w-]+)/)) {
          currentVial.wallet_address = line.match(/^- Wallet Address: ([\w-]+)/)[1];
        }
      }
      if (currentVial) vials.push(currentVial);

      await db.query('UPDATE users SET balance = $1, reputation = $2 WHERE user_id = $3', [wallet.balance, wallet.reputation, user_id]);
      await db.query('DELETE FROM vials WHERE user_id = $1', [user_id]);
      for (const vial of vials) {
        await db.query(
          'INSERT INTO vials (id, user_id, status, balance, wallet_address, created_at) VALUES ($1, $2, $3, $4, $5, $6)',
          [vial.id, user_id, vial.status, vial.balance, vial.wallet_address || wallet.address, new Date()]
        );
      }

      logger.info(`Import completed for user: ${user_id}`);
      return {
        statusCode: 200,
        headers,
        body: JSON.stringify({ message: 'Import successful', vials })
      };
    } else {
      throw new Error('Method not allowed');
    }
  } catch (error) {
    logger.error(`Import/export error: ${error.message}`);
    return {
      statusCode: error.message.includes('Invalid') ? 401 : 500,
      headers,
      body: JSON.stringify({ error: error.message })
    };
  }
};
